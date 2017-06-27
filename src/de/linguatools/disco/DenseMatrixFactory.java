/*******************************************************************************
 *   Copyright (C) 2017 Peter Kolb
 *   peter.kolb@linguatools.org
 *
 *   Licensed under the Apache License, Version 2.0 (the "License"); you may not
 *   use this file except in compliance with the License. You may obtain a copy
 *   of the License at 
 *   
 *        http://www.apache.org/licenses/LICENSE-2.0 
 *
 *   Unless required by applicable law or agreed to in writing, software 
 *   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
 *   License for the specific language governing permissions and limitations
 *   under the License.
 *
 ******************************************************************************/

package de.linguatools.disco;

import static de.linguatools.disco.DenseMatrix.UTF8;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.store.FSDirectory;

/**
 * Creates a <code>DenseMatrix</code> word space from a <code>DISCOLuceneIndex</code>
 * word space.
 * @author peterkolb
 */
public class DenseMatrixFactory {
    
    /**
     * Creates a <code>DenseMatrix</code> object from a <code>DISCOLuceneIndex</code>
     * word space.<br>
     * <b>Warning</b>: use only with low-dimensional (word embedding)
     * word spaces (normally these have less than 1,000 dimensions = features)!
     * @param indexDir <code>DISCOLuceneIndex</code> Lucene index
     * @param numberOfSimilarWords determines how many similar words will be
     * stored for each word, and thereby also determines the <code>WordspaceType</code>
     * of the resulting <code>DenseMatrix</code> object. If you want to create a
     * word space of type <code>COL</code> (that only stores word vectors) then
     * set this to 0.<br/>
     * The maximum value of <code>numberOfSimilarWords</code> is given by the 
     * number of similar words that are stored in the input <code>DISCOLuceneIndex</code>
     * Lucene index. You can look up this value in the <code>disco.config</code>
     * file in <code>indexDir</code>. If <code>indexDir</code> is of type 
     * <code>WordspaceType.COL</code> then this is 0.<br>
     * <b>Note</b>: you can <b>not</b> create a <code>SIM</code> <code>DenseMatrix</code>
     * from a <code>COL</code> <code>DISCOLuceneIndex</code>!
     * @return 
     * @throws IOException
     * @throws FileNotFoundException
     * @throws CorruptConfigFileException 
     */
    public static DenseMatrix create(String indexDir, int numberOfSimilarWords) 
            throws IOException, FileNotFoundException, CorruptConfigFileException{
    
        // word space: stores word vectors for all words
        float[][] matrix; 
        // the optional SIM part:
        int[][] simMatrix; // stores most similar words for each word
        float[][] simValues; // stores the similarity values for the simMatrix
        // other stuff
        Map<String,Integer> word2idMap; // word --> ID (=row number in matrix)
        int[] wordId2offset; // word ID --> offset in offset2word
        int[] frequencies; // word ID --> word frequency in corpus
        byte[] offset2word;
        ConfigFile config;

        // a DenseMatrix is of type COL if it only stores word vectors. In this case
        // simMatrix and simValues both are null.
        // If a DenseMatrix is of type SIM, then the numberOfSimilarWords most 
        // similar words for each word are stored in simMatrix, and the corresponding
        // similarity values in simValues.
        DISCO.WordspaceType wordspaceType;
       
        config = new ConfigFile(indexDir);
        // You can't create a SIM DenseMatrix out of a COL word space. Going
        // from Lucene index to DenseMatrix is only a data structure conversion.
        // If you want to create a SIM DenseMatrix, you need a SIM word space as
        // input. You can create one using DISCOBuilder.
        if( config.dontCompute2ndOrder == true && numberOfSimilarWords != 0 ){
            throw new RuntimeException("The word space in "+indexDir+" is of type"
                    + " DISCO.WordspaceType.COL, therefore numberOfSimilarWords "
                    + " has to be 0. (There are no similar words stored in the index.)");
        }
        
        matrix = new float[config.vocabularySize][config.numberFeatureWords];
        frequencies = new int[config.vocabularySize];
        word2idMap = new HashMap<>();
        
        if( numberOfSimilarWords == 0 ){
            simMatrix = null;
            simValues = null;
            wordspaceType = DISCOLuceneIndex.WordspaceType.COL;
        }else{
            simMatrix = new int[config.vocabularySize][numberOfSimilarWords];
            simValues = new float[config.vocabularySize][numberOfSimilarWords];
            wordspaceType = DISCOLuceneIndex.WordspaceType.SIM;
        }
        
        IndexReader ir = DirectoryReader.open(FSDirectory.open(Paths.get(indexDir)));
        
        // first run over all documents in the index to get a map of all words
        System.out.println("build integer mapping of words in vocabulary...");
        for(int i = 0; i < config.vocabularySize; i++){
            Document doc;
            try {
                doc = ir.document(i);
            } catch (CorruptIndexException ex) {
                continue;
            } catch (IOException ex) {
                continue;
            }
            // Wort Nr. i holen
            String word = doc.get("word");
            // mapping
            if( !word2idMap.containsKey(word) ){
                word2idMap.put(word, word2idMap.size());
            }
            // also get the similar words
            if( wordspaceType == DISCOLuceneIndex.WordspaceType.SIM ){
                String dsb = doc.get("dsb");
                if( dsb != null ){
                    String[] similarWords = dsb.split(" ");
                    for( int w = 0; w < similarWords.length; w++ ){
                        if( w >= numberOfSimilarWords ){
                            break;
                        }
                        if( !word2idMap.containsKey(similarWords[w]) ){
                            word2idMap.put(similarWords[w], word2idMap.size());
                        }
                    }
                }
            }
        }
        System.out.println(word2idMap.size()+" words in vocabulary.");
        
        // count size of all words in bytes plus 2 bytes for each word for
        // storing its length
        int b = 0; int bMax = 0; 
        for( String word : word2idMap.keySet() ){
            int wordBytesLength = word.getBytes(UTF8).length;
            b += wordBytesLength;
            b += 2;
            if( wordBytesLength > bMax ){
                bMax = wordBytesLength;
            }
        }
        System.out.println("\nall words need "+b+" bytes to store in byte array. bMax="+bMax);
        
        // second run
        System.out.println("create dense matrix...");
        int simMax = 0;
        for( int i = 0; i < config.vocabularySize; i++){
            Document doc;
            try {
                doc = ir.document(i);
            } catch (CorruptIndexException ex) {
                continue;
            } catch (IOException ex) {
                continue;
            }
            // Wort Nr. i holen
            String word = doc.get("word");
            // The input index is low-dimensional and has latent variables or IDs
            // as features (not words) and only a fixed number of dimensions
            // (features) for each word. Therefore, we actually do not need to
            // look at the actual "words" in the kol field (because they're just
            // numbers anyway).
            String[] wordsBuffer = doc.get("kol").split(" ");
            String[] valuesBuffer = doc.get("kolSig").split(" ");
            if( wordsBuffer.length != config.numberFeatureWords ){
                System.out.println("*** i="+i+": "+word+": kol.words.length="+wordsBuffer.length
                            +", numberFeatureWords="+config.numberFeatureWords);
            }
            if( valuesBuffer.length != config.numberFeatureWords ){
                System.out.println("*** i="+i+": "+word+": kol.values.length="+valuesBuffer.length
                            +", numberFeatureWords="+config.numberFeatureWords);
            }
            
            // in matrix speichern
            for(int k = 0; k < valuesBuffer.length; k++){
                matrix[i][k] = Float.parseFloat(valuesBuffer[k]);
            }
            // frequency
            frequencies[i] = Integer.parseInt( doc.get("freq") );
            // SIM:
            if( wordspaceType == DISCOLuceneIndex.WordspaceType.SIM ){
                ReturnDataBN res = new ReturnDataBN();
                String dsb = doc.get("dsb");
                if( dsb != null ){
                    res.words = dsb.split(" ");
                    res.values = new float[ res.words.length ];
                    String[] mBuffer = doc.get("dsbSim").split(" ");
                    for( int m = 0; m < mBuffer.length; m++ ){
                        if( m >= res.values.length ){
                            break;
                        }
                        res.values[m] = Float.parseFloat( mBuffer[m] );
                    }
                    for( int s = 0; s < res.words.length; s++ ){
                        if( s >= numberOfSimilarWords ){
                            break;
                        }
                        // here we need to know the ID of the similar word. It
                        // should be among the words in word2idMap.
                        if( !word2idMap.containsKey(res.words[s]) ){
                            throw new RuntimeException("The similar word \""+res.words[s]
                                    +"\" for word \""+word+"\" is not in the word2idMap.");
                        }
                        simMatrix[i][s] = word2idMap.get(res.words[s]);
                        simValues[i][s] = res.values[s];
                    }
                    if( res.words.length > simMax ){
                        simMax = res.words.length;
                    }
                }
            }
            // Info ausgeben
            if( i % 100 == 0 ){
                System.out.print("\r"+i);
            }
        }
        ir.close();
        if( numberOfSimilarWords > simMax ){
            System.out.println("Hint: You set numberOfSimilarWords = "+numberOfSimilarWords
                    +", but the maximum number of similar words in the input index was only "
                    +simMax+". Re-run DenseMatrix construction with numberOfSimilarWords = "
                    +simMax+" in order to get a DenseMatrix with a smaller memory footprint.");
        }
        
        wordId2offset = new int[word2idMap.size()];
        offset2word = new byte[b];
        int offset = 0;
        for( String word : word2idMap.keySet() ){
            byte[] wordBytes = word.getBytes(UTF8);
            System.arraycopy(wordBytes, 0, offset2word, offset + 2, wordBytes.length);
            offset2word[offset] = (byte) (wordBytes.length & 0xFF);
            offset2word[offset+1] = (byte) ((wordBytes.length >> 8) & 0xFF);
            wordId2offset[ word2idMap.get(word) ] = offset;
            offset += wordBytes.length + 2;
        }
    
        return new DenseMatrix(matrix, simMatrix, simValues, word2idMap, wordId2offset,
                frequencies, offset2word, config, wordspaceType, numberOfSimilarWords);
    }

    /**
     * Serialize <code>DenseMatrix</code> object to file.
     * @param denseMatrix
     * @param outputPath 
     */
    public static void serialize(DenseMatrix denseMatrix, String outputPath){
        
        try(
            OutputStream file = new FileOutputStream(outputPath);
            OutputStream buffer = new BufferedOutputStream(file);
            ObjectOutput output = new ObjectOutputStream(buffer);
        ){
            output.writeObject(denseMatrix);
        }catch(IOException ex){
            System.err.println("Cannot serialize DenseMatrix: "+ex);
        }
    }
    
    /**
     * Deserialize <code>DenseMatrix</code> object from file.    
     * @param serializedDenseMatrixPath
     * @return 
     */
    public static DenseMatrix load(File serializedDenseMatrixPath){
        
        DenseMatrix denseMatrix = null;
        
        try(
            InputStream file = new FileInputStream(serializedDenseMatrixPath);
            InputStream buffer = new BufferedInputStream(file);
            ObjectInput input = new ObjectInputStream (buffer);
          ){
            denseMatrix = (DenseMatrix) input.readObject();
        }catch(ClassNotFoundException | IOException ex){
            System.err.println("Cannot deserialize DenseMatrix: "+ex);
        }
        return denseMatrix;
    }

    /**
     * CLI to create <code>DenseMatrix</code> from Lucene index.
     * @param args
     * @throws IOException
     * @throws FileNotFoundException
     * @throws CorruptConfigFileException 
     */
    public static void main(String[] args) throws IOException, FileNotFoundException,
            CorruptConfigFileException{
        
        if( args.length < 2 ){
            System.out.println("*** Create DenseMatrix from DISCOLuceneIndex ***");
            System.out.println("You have to provide the following parameters:");
            System.out.println("DISCOLuceneIndex  serializedOutput  [numberOfSimilarWords]");
            System.out.println("with DISCOLuceneIndex:    word space directory in DISCOLuceneIndex"
                    + " format");
            System.out.println("     numberOfSimilarWords:  number of similar words to store in"
                    + " the output DenseMatrix word space. Allowed values: 0 - numberOfSimilarWords"
                    + " in disco.config file in input."
                    + " 0 will produce a word space of type COL."
                    + " Note that you can not create a SIM word space from a COL word space."
                    + " The default is to use numberOfSimilarWords from disco.config file in input.");
            return;
        }
        
        int numberOfSimilarWords;
        if( args.length == 3 ){
            numberOfSimilarWords = Integer.parseInt(args[2]);
        }else{
            ConfigFile configFile = new ConfigFile(args[0]);
            numberOfSimilarWords = configFile.numberOfSimilarWords;
        }
        
        // create matrix from Lucene index and save serialized matrix
        DenseMatrix matrix = DenseMatrixFactory.create(args[0], numberOfSimilarWords);
        DenseMatrixFactory.serialize(matrix, args[1]);
    }
}
