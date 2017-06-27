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

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.apache.commons.lang3.ArrayUtils;

/**
 * This stores a word space in a dense matrix. Use for low-dimensional word 
 * embeddings only.
 * @author peter
 */
public class DenseMatrix extends DISCO implements Serializable {
    
    // word space: stores word vectors for all words
    private final float[][] matrix; 
    // the optional SIM part:
    private final int[][] simMatrix; // stores most similar words for each word
    private final float[][] simValues; // stores the similarity values for the simMatrix
    // other stuff
    Map<String,Integer> word2idMap; // word --> ID (=row number in matrix)
    private final int[] wordId2offset; // word ID --> offset in offset2word
    private final int[] frequencies; // word ID --> word frequency in corpus
    private final byte[] offset2word;
    private final ConfigFile config;
    
    // a DenseMatrix is of type COL if it only stores word vectors. In this case
    // simMatrix and simValues both are null.
    // If a DenseMatrix is of type SIM, then the numberOfSimilarWords most 
    // similar words for each word are stored in simMatrix, and the corresponding
    // similarity values in simValues.
    private final DISCO.WordspaceType wordspaceType;
    private final int numberOfSimilarWords; // = 0 if wordspaceType == COL
    
    public static final Charset UTF8 = Charset.forName( "UTF-8" );
    private static final long serialVersionUID = 20170125L;
    
    /**
     * Constructor to be used by <code>DenseMatrixFactory</code> only. To create
     * a <code>DenseMatrix</code> from a Lucene index word space use 
     * <code>DenseMatrixFactory.create</code> (or the command line interface). To
     * load a serialized <code>DenseMatrix</code> use <code>DenseMatrixFactory.load</code>.
     * @param matrix
     * @param simMatrix
     * @param simValues
     * @param word2idMap
     * @param wordId2offset
     * @param frequencies
     * @param offset2word
     * @param config
     * @param wordspaceType
     * @param numberOfSimilarWords 
     */
    DenseMatrix(float[][] matrix, int[][] simMatrix, float[][] simValues, 
            Map<String,Integer> word2idMap, int[] wordId2offset, int[] frequencies,
            byte[] offset2word, ConfigFile config, DISCO.WordspaceType wordspaceType, 
            int numberOfSimilarWords){
        
        this.matrix = matrix;
        this.simMatrix = simMatrix;
        this.simValues = simValues;
        this.word2idMap = word2idMap;
        this.wordId2offset = wordId2offset;
        this.frequencies = frequencies;
        this.offset2word = offset2word;
        this.config = config;
        this.wordspaceType = wordspaceType;
        this.numberOfSimilarWords = numberOfSimilarWords;
    }
    
    /**
     * Returns the type of the word space instance.
     * @return word space type
     */
    @Override
    public WordspaceType getWordspaceType(){
        
        return wordspaceType;
    }
    
    /**
     * 
     * @return vocabulary size.
     * @throws IOException 
     */
    @Override
    public int numberOfWords() throws IOException{
        
        return config.vocabularySize;
    }
    
    /**
     * 
     * @return dimensionality of the word space. 
     */
    public int getNumberOfFeatureWords(){
        
        return config.numberFeatureWords;
    }
    
    /**
     * 
     * @param word
     * @return frequency of <code>word</code> in corpus.
     * @throws IOException 
     */
    @Override
    public int frequency(String word) throws IOException{
        
        if( word2idMap.containsKey(word) ){
            return frequencies[ word2idMap.get(word) ];
        }else{
            return 0;
        }
    }
    
    /**
     * Only works with word spaces of type <code>DISCO.WordspaceType.SIM</code>.
     * @param word
     * @return list of similar words for <code>word</code> if these are stored
     * in the word space.
     * @throws IOException
     * @throws WrongWordspaceTypeException 
     */
    @Override
    public ReturnDataBN similarWords(String word) throws IOException, 
            WrongWordspaceTypeException{
        
        // check word space type
        if( wordspaceType != DISCO.WordspaceType.SIM ){
            throw new WrongWordspaceTypeException("This method can not be applied"
                    + " to word spaces of type "+wordspaceType);
        }
        
        int id;
        if( word2idMap.containsKey(word) ){
            id = word2idMap.get(word);
        }else{
            return null;
        }
        
        List<String> dsb = new LinkedList<>();
        List<Float> dsbSim = new LinkedList<>();
        for(int i = 0; i < numberOfSimilarWords; i++ ){
            if( simValues[id][i] == 0 ){
                break;
            }
            dsb.add( id2word(simMatrix[id][i]) );
            dsbSim.add( simValues[id][i] );
        }
        
        // create return object
        ReturnDataBN res = new ReturnDataBN();
        res.words = new String[ dsb.size() ];
        res.values = new float[ dsbSim.size() ];
        for( int i = 0; i < dsb.size(); i++ ){
            res.words[i] = dsb.get(i);
            res.values[i] = dsbSim.get(i);
        }
        return res;
    }
    
    private float computeSimilarityKolb(int i1, int i2){
        
        float nenner = 0;
        float zaehler = 0;
        for( int k = 0; k < config.numberFeatureWords; k++ ){
            nenner += matrix[i1][k] + matrix[i2][k];
            if( matrix[i1][k] > 0 && matrix[i2][k] > 0 ){
                zaehler += matrix[i1][k] + matrix[i2][k];
            }
        }
        return 2 * zaehler / nenner;  // DICE-KOEFFIZIENT !
    }
    
    private float computeSimilarityCosine(int i1, int i2){
        
        float nenner1 = 0;
        float nenner2 = 0;
        float zaehler = 0;
        for( int k = 0; k < config.numberFeatureWords; k++ ){
            nenner1 += matrix[i1][k] * matrix[i1][k];
            nenner2 += matrix[i2][k] * matrix[i2][k];
            zaehler += matrix[i1][k] * matrix[i2][k];
        }
        return (float) (zaehler / Math.sqrt(nenner1 * nenner2));
    }
    
    @Override
    public float semanticSimilarity(String w1, String w2, SimilarityMeasure 
            similarityMeasure) throws IOException{
        
        // die beiden zu vergleichenden Wörter im Trie nachschlagen
        int i1;
        if( word2idMap.containsKey(w1) ){
            i1 = word2idMap.get(w1);
        }else{
            return -2;
        }
        int i2;
        if( word2idMap.containsKey(w2) ){
            i2 = word2idMap.get(w2);
        }else{
            return -2;
        }
        
        if( similarityMeasure == SimilarityMeasure.KOLB ){
            // Vektorähnlichkeitsmaß "Kolb" gibt Wert zwischen 0 und 1 zurück
            return computeSimilarityKolb(i1, i2);
        }else if( similarityMeasure == SimilarityMeasure.COSINE ){
            // Vektorähnlichkeitsmaß "Kosinus" gibt Wert zwischen -1 und 1 zurück
            return computeSimilarityCosine(i1, i2);
        }else{
            // unknown similarity measure
            return -3.0F;
        }
    }
    
    @Override
    public float secondOrderSimilarity(String w1, String w2)
            throws IOException, WrongWordspaceTypeException{
        
        // check word space type
        if( wordspaceType != DISCOLuceneIndex.WordspaceType.SIM ){
            throw new WrongWordspaceTypeException("This method can not be applied"
                    + "to word spaces of type "+wordspaceType);
        }
        
        // die beiden zu vergleichenden Wörter im Trie nachschlagen
        int i1;
        if( word2idMap.containsKey(w1) ){
            i1 = word2idMap.get(w1);
        }else{
            return -2;
        }
        int i2;
        if( word2idMap.containsKey(w2) ){
            i2 = word2idMap.get(w2);
        }else{
            return -2;
        }
        
        float nenner = 0;
        HashMap<Integer,Float> simHash = new HashMap();
        for(int i = 0; i < numberOfSimilarWords; i++ ){
            if( simValues[i1][i] > 0 ){
                simHash.put(simMatrix[i1][i], simValues[i1][i]);
                nenner += simValues[i1][i];
            }else{
                break;
            }
        }
        float zaehler = 0;
        for(int i = 0; i < numberOfSimilarWords; i++ ){
            if( simValues[i2][i] > 0 ){
                if ( simHash.containsKey(simMatrix[i2][i]) ){
                    zaehler += simValues[i2][i] + simHash.get(simMatrix[i2][i]);
                }
                nenner += simValues[i2][i];
            }else{
                break;
            }
        }
        return 2 * zaehler / nenner; // x 2 ???
    }
    
    /**
     * The word vector in a dense matrix contains only IDs as keys.
     * @param word
     * @return
     * @throws IOException 
     */
    @Override
    public HashMap<String,Float> getWordvector(String word)
            throws IOException{

        int id;
        if( word2idMap.containsKey(word) ){
            id = word2idMap.get(word);
        }else{
            return null;
        }

        HashMap<String,Float> wv = new HashMap<>();
        for( int i = 0; i < config.numberFeatureWords; i++ ){
            wv.put(String.valueOf(i), matrix[id][i]);
        }
        
        return wv;
    }
    
    @Override
    public int wordFrequencyList(String outputFileName){
     
        try {
            // öffne Ausgabedatei
            FileWriter fw = new FileWriter(outputFileName);
            for( int i = 0; i < frequencies.length; i++ ){
                // Wort und Frequenz in Ausgabe schreiben
                fw.write( id2word(i)+"\t"+frequencies[i]+"\n");
            }
            fw.close();
        } catch (IOException ex) {
            System.out.println(DenseMatrix.class.getName()+": "+ex);
            return -1;
        }
        return frequencies.length;
    }
    
    @Override
    public String[] getStopwords() throws FileNotFoundException, IOException,
            CorruptConfigFileException{
        
        return config.stopwords.trim().split("\\s+");
    }
    
    @Override
    public long getTokenCount(){
        
        return config.tokencount;
    }
    
    @Override
    public int getMinFreq(){
        
        return config.minFreq;
    }
    
    @Override
    public int getMaxFreq(){
        
        return config.maxFreq;
    }
    
    @Override
    public Iterator<String> getVocabularyIterator(){
        
        return new VocabularyIterator();
    }
    
    class VocabularyIterator implements Iterator<String>{
        
        private int i;
        
        public VocabularyIterator(){
            
            i = 0;
        }
        
        @Override
        public boolean hasNext(){
            
            if( i < config.vocabularySize ){
                return true;
            }else{
                return false;
            }
        }
        
        @Override
        public String next(){
            
            int buffer = i;
            i++;
            return id2word( buffer );
        }
        
        @Override
        public void remove(){
            
        }
    }
    
    @Override
    public String getWord(int id) throws IOException{
        
        if( id >= config.vocabularySize ){
            return null;
        }
        return id2word(id);
    }
    
    ////////////////////////////////////////////////////////////////////////////
    // additional methods specific to DenseMatrix
    ////////////////////////////////////////////////////////////////////////////
    
    private String id2word(int id){
        
        int offset = wordId2offset[ id ];
        
        int wordBytesSize = (int) offset2word[offset] & 0xFF;
        wordBytesSize += (int) (offset2word[offset+1] & 0xFF) << 8;
        String word = new String(
                ArrayUtils.subarray(offset2word, offset + 2, offset + 2 + wordBytesSize),
                UTF8);
        return word;
    }
    
    public float similarityCosine(String w1, String w2){
        
        int i1;
        if( word2idMap.containsKey(w1) ){
            i1 = word2idMap.get(w1);
        }else{
            return -2;
        }
        int i2;
        if( word2idMap.containsKey(w2) ){
            i2 = word2idMap.get(w2);
        }else{
            return -2;
        }
        
        return similarityCosine(i1, i2);
    }
    
    public float similarityCosine(int i1, int i2){
        
        float zaehler = 0, nenner1 = 0, nenner2 = 0;
        for(int k = 0; k < config.numberFeatureWords; k++){
            nenner1 += matrix[i1][k] * matrix[i1][k];
            nenner2 += matrix[i2][k] * matrix[i2][k];
            zaehler += matrix[i1][k] * matrix[i2][k];
        }
        return (float) (zaehler / Math.sqrt(nenner1 * nenner2));
    }
    
    public List<ReturnDataCol> getMostSimilar(int i, int max){
        
        List<ReturnDataCol> similarWords = new ArrayList<>();
        
        for( int k = 0; k < config.vocabularySize; k++ ){
            if( k == i ){
                continue;
            }
            float sim = similarityCosine(i, k);
            if( sim <= 0.0F ){
                continue;
            }
            similarWords.add(new ReturnDataCol( id2word(k), sim));
        }
        
        Collections.sort(similarWords);
        
        if( similarWords.size() < max ){
            max = similarWords.size();
        }
        return similarWords.subList(0, max);
    }
    
    public void printMostSimilar(String w, int max){
        
        int i;
        if( word2idMap.containsKey(w) ){
            i = word2idMap.get(w);
        }else{
            return;
        }
        
        List<ReturnDataCol> similarWords = new ArrayList<>();
        
        for( int k = 0; k < config.vocabularySize; k++ ){
            if( k == i ){
                continue;
            }
            float sim = similarityCosine(i, k);
            if( sim <= 0.0F ){
                continue;
            }
            similarWords.add(new ReturnDataCol( id2word(k), sim));
        }
        
        Collections.sort(similarWords);
        int out = 0;
        for( ReturnDataCol data : similarWords ){
            System.out.println(data.word+"\t"+data.value);
            out++;
            if( out >= max ){
                break;
            }
        }
    }
    
    /**
     * Returns the matrix row for the word, i.e. the dense word vector. 
     * @param word
     * @return <code>null</code> if <code>word</code> is not found.
     */
    public float[] getVector(String word){
        
        int id;
        if( word2idMap.containsKey(word) ){
            id = word2idMap.get(word);
        }else{
            return null;
        }
        
        return matrix[id];
    }
    
    
}
