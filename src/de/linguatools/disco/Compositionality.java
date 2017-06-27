/*******************************************************************************
 *   Copyright (C) 2007, 2008, 2009, 2010, 2011, 2012, 2015, 2016 Peter Kolb
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

import de.linguatools.disco.DISCO.SimilarityMeasure;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * This class provides support for compositional distributional semantics.
 * There are methods to compute the similarity between multi-word terms,
 * phrases and sentences or even paragraphs based on composition of the vectors
 * of individual words.
 * @author peter
 * @version 2.0
 */
public class Compositionality {
    
    /**
     * Implemented methods of vector composition.
     */
    public enum VectorCompositionMethod {
        /**
         * Simple vector addition.
         */
        ADDITION, 
       /**
        * Vector subtraction.
        */
        SUBTRACTION,
        /**
         * Entry-wise multiplication.
         */
        MULTIPLICATION, 
        /**
         * Parameterized combination of addition and multiplication, cf. 
         * equation (11) in J. Mitchell and M. Lapata: Vector-based Models of
         * Semantic Composition. Proceedings of ACL-08: HLT.
         */
        COMBINED, 
        /**
         * Dilate word vector u along the direction of word vector v: 
         *  <blockquote>v' = u ° v<br/>
         *     = (lambda-1)(u*v/u*u)*u + v</blockquote>
         * If SimilarityMeasures.COSINE is used, the following formula can be
         * used instead:
         *  <blockquote>v' = (u*u)v + (lambda-1)(u*v)u</blockquote>
         * where * is the dot product (Skalarprodukt).<br/>
         * Contrary to the other composition methods, this operation is not
         * symmetric.<br/>
         * See chapter 4 of J. Mitchell: Composition in Distributional Models of
         * Semantics. PhD, Edinburgh, 2011.
         */
        DILATION,
        /**
         * Vector extrema combines vectors by choosing for each vector dimension
         * the value that has the highest distance from zero, i.e. the highest 
         * absolute value.
         * See G. Forgues et al. (2014): Bootstrapping Dialog Systems with Word
         * Embeddings. NIPS 2014.
         */
        EXTREMA;
    }
    
    /**
     * Compute the dot product (inner product, scalar product) of wv1 and wv2.
     * @param wv1 first word vector
     * @param wv2 second word vector
     * @return result (a scalar, not a vector)
     */
    private static float computeDotProduct(HashMap<String,Float> wv1, 
            HashMap<String,Float> wv2){
        
        float sp = 0.0F;
        for (String w : wv1.keySet()) {
            if( wv2.containsKey(w) ){
                sp = sp + wv1.get(w) * wv2.get(w);
            }
        }
        return sp;
    }
    
    /**
     * The following formula is used:
     * <blockquote>(wv1*wv1)wv2 + (lambda-1)(wv1*wv2)wv1</blockquote>
     * The default value (if lambda is null) for lambda is 2.0.<br/>
     * This composition method only works with the SimilarityMeasures.COSINE
     * similarity measure. 
     * @param wv1
     * @param wv2
     * @param lambda
     * @return 
     */
    private static HashMap<String,Float> composeVectorsByDilation(
            HashMap<String,Float> wv1, HashMap<String,Float> wv2, Float lambda){
        
        if( lambda == null) lambda = 2.0F;
        
        float a = computeDotProduct(wv1, wv2);
        HashMap<String,Float> f1 = multiplicateWordVectorWithScalar(wv2, a);
        HashMap<String,Float> f2 = multiplicateWordVectorWithScalar(wv1, 
                a*(lambda-1));
        return composeVectorsByAddition(f1, f2);
    }
    
    /**
     * Multiply all values in the word vector hash with the scalar. 
     * @param wv word vector
     * @param scalar
     * @return 
     */
    private static HashMap<String,Float> multiplicateWordVectorWithScalar(
            HashMap<String,Float> wv, float scalar){
        
        for (String w : wv.keySet()) {
            wv.put(w, wv.get(w) * scalar);
        }
        return wv;
    }
    
    /**
     * Compose vectors wv1 and wv2 by a combination of addition and 
     * multiplication:
     * <blockquote>p = a*wv1 + b*wv2 + c*wv1*wv2</blockquote>
     * The contribution of multiplication and addition, as well
     * as the contribution of each of the two vectors can be controlled by the
     * three parameters a, b and c.<br/>
     * For instance, in Mitchell and Lapata 2008 where wv1 is a verb and wv2 is
     * a noun, the parameters a, b and c are set as follows:
     * <blockquote>a = 0.95<br/>
     * b = 0<br/>
     * c = 0.05.</blockquote>
     * If one of a, b, c is null, then these default values are used.
     * @param wv1 first word vector
     * @param wv2 second word vector
     * @param a weight of additive contribution of first word vector
     * @param b weight of additive contribution of second word vector
     * @param c weight of multiplicative contribution of both word vectors
     * @return 
     */
    private static HashMap<String,Float> composeVectorsByCombinedMultAdd(
            HashMap<String,Float> wv1, HashMap<String,Float> wv2, Float a, 
            Float b, Float c){
        
        if( a == null || b == null || c == null ){
            a = 0.95F;
            b = 0.0F;
            c = 0.05F;
        }
        
        // Formula: result = a*wv1 + b*wv2 + c*wv1*wv2
        // m = wv1 * wv2
        HashMap<String,Float> m = composeVectorsByMultiplication(wv1, wv2);
        // m = c * m
        m = multiplicateWordVectorWithScalar(m, c);
        // k = a * wv1
        HashMap<String,Float> k = multiplicateWordVectorWithScalar(wv1, a);
        // l = b * wv2
        HashMap<String,Float> l = multiplicateWordVectorWithScalar(wv2, b);
        // result = k + l + m
        return composeVectorsByAddition(composeVectorsByAddition(k,l),m);
    }
    
    /**
     * Combines two word vectors by multiplication.
     * @param wv1 word vector #1
     * @param wv2 word vector #2
     * @return the combined word vector
     */
    private static HashMap<String,Float> composeVectorsByMultiplication(
            HashMap<String,Float> wv1, HashMap<String,Float> wv2){
        
        HashMap<String,Float> result = new HashMap();
        for (String feature : wv1.keySet()) {
            if( wv2.containsKey(feature) ){
                result.put(feature, wv1.get(feature) * wv2.get(feature));
            }
        }
        return result;
    }
    
    /**
     * Combines two word vectors by addition.
     * @param wv1
     * @param wv2
     * @return the combined word vector
     */
    private static HashMap<String,Float> composeVectorsByAddition(
            HashMap<String,Float> wv1, HashMap<String,Float> wv2){
        
        HashMap<String,Float> result = new HashMap();
        for (String w : wv1.keySet()) {
            if( !wv2.containsKey(w) ){
                result.put(w, wv1.get(w));
            }
        }
        for (String w : wv2.keySet()) {
            if( wv1.containsKey(w) ){
                result.put(w, wv1.get(w) + wv2.get(w));
            }else{
                result.put(w, wv2.get(w));
            }
        }
        
        return result;
    }
    
    /**
     * Subtract the second word vector from the first: wv1 - wv2.
     * @param wv1
     * @param wv2
     * @return the combined word vector
     */
    private static HashMap<String,Float> composeVectorsBySubtraction(
            HashMap<String,Float> wv1, HashMap<String,Float> wv2){
        
        HashMap<String,Float> result = new HashMap();
        for (String w : wv1.keySet()) {
            if( !wv2.containsKey(w) ){
                result.put(w, wv1.get(w));
            }else{
                result.put(w, wv1.get(w) - wv2.get(w));
            }
        }
        for (String w : wv2.keySet()) {
            if( !wv1.containsKey(w) ){
                result.put(w, -wv2.get(w));
            }
        }
        
        return result;
    }
    
    /**
     * Choose for each dimension the highest absolute value.
     * @param wv1
     * @param wv2
     * @return 
     */
    private static HashMap<String,Float> composeVectorsByExtrema(
            HashMap<String,Float> wv1, HashMap<String,Float> wv2){
        
        HashMap<String,Float> result = new HashMap();
        for (String w : wv1.keySet()) {
            if( !wv2.containsKey(w) ){
                result.put(w, wv1.get(w));
            }else{
                if( Math.abs(wv1.get(w)) >= Math.abs(wv2.get(w)) ){
                    result.put(w, wv1.get(w));
                }else{
                    result.put(w, wv2.get(w));
                }
            }
        }
        for (String w : wv2.keySet()) {
            if( !wv1.containsKey(w) ){
                result.put(w, wv2.get(w));
            }
        }
        
        return result;
    }
    
    /**
     * Compute the average vector of all vectors in the list.
     * @param vectors
     * @return average vector
     * @since 3.0
     */
    public static HashMap<String,Float> averageVector(List<HashMap<String,Float>> vectors){
        
        HashMap<String,Float> result = new HashMap();
        
        // sum up for all dimensions
        for( HashMap<String,Float> v : vectors ){
            for( String w : v.keySet() ){
                if( result.containsKey(w) ){
                    result.put(w, v.get(w) + result.get(w));
                }else{
                    result.put(w, v.get(w));
                }
            }
        }
        // divide each dimension's value by number of vectors
        for( String w : result.keySet() ){
            result.put(w, (float) result.get(w) / (float)vectors.size());
        }
        return result;
    }
    
    /**
     * Computes vector rejection of a on b. See https://en.wikipedia.org/wiki/Vector_projection
     * and http://bookworm.benschmidt.org/posts/2015-10-25-Word-Embeddings.html.<br/>
     * Example: to get the "river" meaning for the word "bank" use vector rejection:
     * bank_without_finance = vectorRejection(bank, averageVector(deposit, account, cashier)).
     * @param a 
     * @param b
     * @return 
     * @since 3.0
     */
    public static HashMap<String,Float> vectorRejection(HashMap<String,Float> a,
            HashMap<String,Float> b){
        
        return composeVectorsBySubtraction(a, multiplicateWordVectorWithScalar(b, 
                computeDotProduct(a, b) / computeDotProduct(b, b)));
    }
    
    /**
     * Compose two word vectors by the composition method given in 
     * <code>compositionMethod</code>.
     * @param wv1 word vector #1
     * @param wv2 word vector #2
     * @param compositionMethod One of the methods in <code>VectorCompositionMethod</code>.
     * @param a only needed for composition method COMBINED.
     * @param b only needed for composition method COMBINED.
     * @param c only needed for composition method COMBINED.
     * @param lambda only needed for composition method DILATION.
     * @return the resulting word vector or <code>null</code>.
     */
    public static HashMap<String,Float> composeWordVectors(HashMap<String,Float> wv1,
            HashMap<String,Float> wv2, VectorCompositionMethod compositionMethod,
            Float a, Float b, Float c, Float lambda){
    
        if( wv1 == null || wv2 == null ){
            return null;
        }
        
        if( compositionMethod == VectorCompositionMethod.ADDITION ){
            return composeVectorsByAddition(wv1, wv2);
        }else if( compositionMethod == VectorCompositionMethod.SUBTRACTION ){
            return composeVectorsBySubtraction(wv1, wv2);
        }else if( compositionMethod == VectorCompositionMethod.MULTIPLICATION ){
            return composeVectorsByMultiplication(wv1, wv2);
        }else if( compositionMethod == VectorCompositionMethod.COMBINED ){
            return composeVectorsByCombinedMultAdd(wv1, wv2, a, b, c); 
        }else if( compositionMethod == VectorCompositionMethod.DILATION ){
            return composeVectorsByDilation(wv1, wv2, lambda);   
        }else if( compositionMethod == VectorCompositionMethod.EXTREMA ){
            return composeVectorsByExtrema(wv1, wv2);     
        }else{
            return null;
        }
    }
    
    /**
     * Compose two or more word vectors by the composition method given in 
     * <code>compositionMethod</code>.
     * @param wordvectorList a list of word vectors to be combined. The list has
     * to have at least two elements. The ordering of the list has no influence
     * on the result.
     * @param compositionMethod One of the methods in <code>VectorCompositionMethod</code>.
     * @param a only needed for composition method COMBINED.
     * @param b only needed for composition method COMBINED.
     * @param c only needed for composition method COMBINED.
     * @param lambda only needed for composition method DILATION.
     * 
     * @return the resulting word vector or <code>null</code>.
     */
    public static HashMap<String,Float> composeWordVectors(ArrayList<HashMap<String,Float>>
            wordvectorList, VectorCompositionMethod compositionMethod, Float a, 
            Float b, Float c, Float lambda){
        
        if( wordvectorList.size() < 2 ){
            return null;
        }
        if( wordvectorList.get(0) == null || wordvectorList.get(1) == null ){
            return null;
        }
        
        // combine the first two vectors in the list
        HashMap<String,Float> wv = composeWordVectors(wordvectorList.get(0),
                wordvectorList.get(1), compositionMethod, a, b, c, lambda);
        
        for(int i = 2; i < wordvectorList.size(); i++){
            if( wordvectorList.get(i) == null ){
                continue;
            }
            wv = composeWordVectors(wv, wordvectorList.get(i), compositionMethod,
                    a, b, c, lambda);
        }
        return wv;
    }
    
    /**
     * Utility function. Prints the word vector to standard output.
     * @param wordvector 
     */
    public static void printWordVector(HashMap<String,Float> wordvector){
        
        for (String w : wordvector.keySet()) {
            System.out.println(w+"\t"+wordvector.get(w));
        }
    }
    
    /**
     * This method compares two word vectors using the similarity measure
     * SimilarityMeasures.KOLB that is described in the paper
     * <blockquote>Peter Kolb. <a href="http://hdl.handle.net/10062/9731">Experiments
     * on the difference between semantic similarity and relatedness</a>. In 
     * <i>Proceedings of the <a href="http://beta.visl.sdu.dk/nodalida2009/">17th
     * Nordic Conference on Computational Linguistics - NODALIDA '09</a></i>, 
     * Odense, Denmark, May 2009.</blockquote>
     * @param wv1 a word vector
     * @param wv2 another word vector
     * @return the similarity between the two word vectors; a value between 0.0F
     * and 1.0F.
     */
    private static float computeSimilarityKolb(HashMap<String,Float> wv1, 
            HashMap<String,Float> wv2){
        
        float nenner = 0;
        for( Iterator it = wv1.keySet().iterator(); it.hasNext(); ){
            nenner += wv1.get( (String) it.next());
        }
        
        float zaehler = 0;
        for (String w : wv2.keySet()) {
            float v = wv2.get(w);
            if ( wv1.containsKey(w) ){
                zaehler += (v + wv1.get(w));
            }
            nenner += v;
        }
        return 2 * zaehler / nenner;  // DICE-KOEFFIZIENT !
    }
    
    /**
     * This method compares two word vectors using the similarity measure
     * SimilarityMeasures.COSINE.
     * @param wv1 a word vector
     * @param wv2 another word vector
     * @return the similarity between the two word vectors; a value between -1.0F
     * and 1.0F. A return value of -2.0F indicates an error.
     */
    private static float computeSimilarityCosine(HashMap<String,Float> wv1, 
            HashMap<String,Float> wv2){
        
        if( wv1 == null || wv2 == null ){
            return -2.0F;
        }
        
        float nenner1 = 0.0F;
        for( Iterator it = wv1.keySet().iterator(); it.hasNext(); ){
            float v = wv1.get( (String) it.next());
            nenner1 += v * v;
        }
        
        float nenner2 = 0, zaehler = 0;
        for (String w : wv2.keySet()) {
            float v = wv2.get(w);
            if ( wv1.containsKey(w) ){
                zaehler += (v * wv1.get(w));
            }
            nenner2 += v * v;
        }
        return (float) (zaehler / Math.sqrt(nenner1 * nenner2));
    }
    
    /**
     * Computes the semantic similarity (according to the vector similarity 
     * measure <code>similarityMeasure</code>) between the two input word 
     * vectors.<br/>
     * @param wordvector1
     * @param wordvector2
     * @param simMeasure One of the similarity measures enumerated in
     * <code>DISCOLuceneIndex.SimilarityMeasures</code>.
     * @return The similarity between the two input word vectors; depending on
     * the chosen similarity measure a value between 0.0F and 1.0F, or -1.0F and 
     * 1.0F. In case the <code>similarityMeasure</code> is unknown the return
     * value is -3.0F.
     */
    public static float semanticSimilarity(HashMap<String,Float> wordvector1, 
            HashMap<String,Float> wordvector2, SimilarityMeasure simMeasure){
        
        if( simMeasure == SimilarityMeasure.KOLB ){
            return computeSimilarityKolb(wordvector1, wordvector2);
        }else if( simMeasure == SimilarityMeasure.COSINE ){
            return computeSimilarityCosine(wordvector1, wordvector2);
        }else{
            return -3.0F;
        }
    }
    
    /***************************************************************************
     * Computes the semantic similarity (according to the vector similarity 
     * measure <code>SimilarityMeasures.KOLB</code> which is described in 
     * <a href="http://hdl.handle.net/10062/9731">Kolb 2009</a>) between the 
     * two input word vectors.
     * @param wordvector1
     * @param wordvector2
     * @return The similarity between the two input word vectors; a value
     * between 0.0F and 1.0F.
     */
    public static float semanticSimilarity(HashMap<String,Float> wordvector1, 
            HashMap<String,Float> wordvector2){
        
        return computeSimilarityKolb(wordvector1, wordvector2);
    }
    
    /**
     * This method computes the semantic similarity between two multi-word terms,
     * phrases, sentences or paragraphs. This is done by composition of the word
     * vectors of the constituent words.<br/>
     * Each of the two input strings is split at whitespace, and the wordvectors
     * of the individual tokens (constituent words) are retrieved. Then the
     * word vectors are combined using the method <code>composeWordVectors()</code>.
     * The two resulting vectors are then compared using
     * <code>Compositionality.semanticSimilarity()</code>.<br/>
     * <b>Note</b>: the methods in class <code>TextSimilarity</code> might give
     * more accurate results for short text similarity because they weight the
     * words in the input strings by their frequency and try to align words in 
     * the input strings.
     * @param multiWords1 a tokenized string containing a multi-word term, phrase,
     * sentence or paragraph.
     * @param multiWords2 a tokenized string containing a multi-word term, phrase,
     * sentence or paragraph.
     * @param compositionMethod a vector composition method.
     * @param simMeasure a similarity measure. 
     * @param disco a DISCOLuceneIndex word space.
     * @param a only needed for composition method COMBINED.
     * @param b only needed for composition method COMBINED.
     * @param c only needed for composition method COMBINED.
     * @param lambda only needed for composition method DILATION.
     * @return the distributional similarity between <code>multiWord1</code> and
     * <code>multiWord2</code>.
     * @throws java.io.IOException
     * @see de.linguatools.disco.TextSimilarity
     */
    public static float compositionalSemanticSimilarity(String multiWords1, 
            String multiWords2, VectorCompositionMethod compositionMethod, 
            SimilarityMeasure simMeasure, DISCOLuceneIndex disco, Float a, 
            Float b, Float c, Float lambda) throws IOException{
        
        multiWords1 = multiWords1.trim();
        multiWords2 = multiWords2.trim();
        String[] multi1 = multiWords1.split("\\s+");
        String[] multi2 = multiWords2.split("\\s+");
        
        // compute word vector #1
        HashMap<String,Float> wv1;
        if( multi1.length == 1 ){
            wv1 = disco.getWordvector(multi1[0]);
        }else if( multi1.length == 2 ){
            wv1 = composeWordVectors(disco.getWordvector(multi1[0]),
                disco.getWordvector(multi1[1]), compositionMethod, a, b, c, lambda);
        }else{
            wv1 = composeWordVectors(disco.getWordvector(multi1[0]),
                disco.getWordvector(multi1[1]), compositionMethod, a, b, c, lambda);
            for(int i = 2; i < multi1.length; i++){
                wv1 = composeWordVectors(wv1, disco.getWordvector(multi1[i]),
                        compositionMethod, a, b, c, lambda);
            }
        }
        
        // compute word vector 21
        HashMap<String,Float> wv2;
        if( multi2.length == 1 ){
            wv2 = disco.getWordvector(multi2[0]);
        }else if( multi2.length == 2 ){
            wv2 = composeWordVectors(disco.getWordvector(multi2[0]),
                disco.getWordvector(multi2[1]), compositionMethod, a, b, c, lambda);
        }else{
            wv2 = composeWordVectors(disco.getWordvector(multi2[0]),
                disco.getWordvector(multi2[1]), compositionMethod, a, b, c, lambda);
            for(int i = 2; i < multi2.length; i++){
                wv2 = composeWordVectors(wv2, disco.getWordvector(multi2[i]),
                        compositionMethod, a, b, c, lambda);
            }
        }
        
        // compute similarity between the two word vectors
        return semanticSimilarity(wv1, wv2, simMeasure);
    }
    
    /**
     * Find the most similar words in the DISCO word space for an input word 
     * vector. While the word vector can represent a multi-token word (if it was
     * produced by one of the methods 
     * <code>Compositionality.composeWordVectors()</code>) the most
     * similar words will only be single-token words from the index.<br/>
     * <b>Warning</b>: This method is very time consuming and should only be
     * used with a word space that has been loaded into memory!
     * @param wordvector input word vector
     * @param disco DISCO word space
     * @param simMeasure
     * @param maxN return only the <code>maxN</code> most similar words. If 
     * <code>maxN &lt; 1</code> all words are returned. 
     * @return List of all words (with their similarity values) whose similarity
     * with the <code>wordvector</code> is greater than zero, ordered by 
     * similarity value (highest value first).
     * @throws IOException 
     */
    public static List<ReturnDataCol> similarWords(HashMap<String,Float> wordvector,
            DISCO disco, SimilarityMeasure simMeasure, int maxN)
            throws IOException{
        
        List<ReturnDataCol> result = new ArrayList();
        
        // durchlaufe alle Dokumente
        Iterator<String> iterator = disco.getVocabularyIterator();
        while( iterator.hasNext() ){
            String word = iterator.next();
            HashMap<String,Float> wv = disco.getWordvector(word);
            // Ähnlichkeit zwischen Wortvektoren berechnen
            float sim = semanticSimilarity(wordvector, wv, simMeasure);
            if( sim > 0.0F){
                ReturnDataCol r = new ReturnDataCol(word, sim);
                result.add(r);
            }
        }
        
        // nach höchstem Ähnlichkeitswert sortieren
        Collections.sort(result);
        if( maxN > 0 ){
            result = result.subList(0, maxN);
        }
        return result;
    }
    
    /**
     * Experimental!
     * @param wordvector
     * @param disco
     * @param simMeasure
     * @param maxN
     * @return
     * @throws IOException
     * @throws WrongWordspaceTypeException 
     */
    public static List<ReturnDataCol> similarWordsGraphSearch(HashMap<String,Float> wordvector,
            DISCO disco, SimilarityMeasure simMeasure, int maxN)
            throws IOException, WrongWordspaceTypeException{
        
        // check word space type
        if( disco.getWordspaceType() != DISCO.WordspaceType.SIM ){
            throw new WrongWordspaceTypeException("This method can not be applied"
                    + "to word spaces of type "+disco.getWordspaceType());
        }
        
        List<ReturnDataCol> result = new ArrayList();
        
        // pick random start word
        int start = ThreadLocalRandom.current().nextInt(0, disco.numberOfWords() );
        String startWord = disco.getWord(start);
        HashMap<String,Float> wvStart = disco.getWordvector(startWord);
        float sim = semanticSimilarity(wvStart, wordvector, simMeasure);
        boolean better = false;
        do{
            // get similar words for start word
            ReturnDataBN similarWords = disco.similarWords(startWord);
            // find the most similar word to the input word vector. Only look at the
            // first depth words
            int depth = 20;
            String maxWord = startWord;
            float maxSim = sim;
            HashMap<String,Float> maxWordvector = wvStart;
            for( int i = 0; i < similarWords.words.length; i++ ){
                if( i == depth ){
                    break;
                }
                HashMap<String,Float> wv = disco.getWordvector(similarWords.words[i]);
                float s = semanticSimilarity(wordvector, wv, simMeasure);
                if( s > maxSim ){
                    maxSim = s;
                    maxWord = similarWords.words[i];
                    maxWordvector = wv;
                }
            }
            if( maxSim > sim ){
                sim = maxSim;
                startWord = maxWord;
                wvStart = maxWordvector;
                better = true;
            }
        }while( better );
  
        ReturnDataCol r = new ReturnDataCol(startWord, sim);
        result.add(r);
        // nach höchstem Ähnlichkeitswert sortieren
//        Collections.sort(result);
//        if( maxN > 0 ){
//            result = result.subList(0, maxN);
//        }
        return result;
    }
    
    /**
     * This method solves the analogy "w1 is to w2 like x is to w3", i.e. it 
     * returns the missing word x. Example: "king is to man like x is to woman"
     * with x = queen. This is done by the formula v(x) = v(w1) - v(w2) + v(w3),
     * where v(w) is the word vector for a word w.<br/>
     * The methods vector addition and subtraction from the 
     * <code>Compositionality</code> class are used.<br/>
     * This works best with word spaces computed with word2vec.<br/>
     * <b>Warning:</b> This method is very time consuming because after computing
     * v(x), the most similar word vector to v(x) has to be found in the word 
     * space. This is done by comparing <b>all</b> vectors in the word space with
     * v(x).
     * @param w1 first word (must be single token)
     * @param w2 second word (must be single token)
     * @param w3 third word (must be single token)
     * @param disco
     * @return ordered list with the nearest words to v(x) or null if one of 
     * words w1, w2, or w3 was not found in the DISCO index. You may want to 
     * filter out w1, w2, and w3 from the resulting list.
     * @throws IOException 
     * @throws de.linguatools.disco.WrongWordspaceTypeException 
     */
    public static List<ReturnDataCol> solveAnalogy(String w1, String w2, String w3,
            DISCO disco) throws IOException, WrongWordspaceTypeException{
        
        // get word vectors from DISCO word space
        HashMap<String,Float> wv1 = disco.getWordvector(w1);
        if( wv1 == null ){
            return null;
        }
        HashMap<String,Float> wv2 = disco.getWordvector(w2);
        if( wv2 == null ){
            return null;
        }
        HashMap<String,Float> wv3 = disco.getWordvector(w3);
        if( wv3 == null ){
            return null;
        }
        
        // compute wvx = wv1 - wv2 + wv3
        HashMap<String,Float> temp = composeWordVectors(wv1, wv2,
                VectorCompositionMethod.SUBTRACTION,
                null, null, null, null);
        HashMap<String,Float> wvx = composeWordVectors(temp, wv3,
                VectorCompositionMethod.ADDITION,
                null, null, null, null);
        
        // find nearest words for wvx
        return similarWords(wvx, disco, SimilarityMeasure.COSINE, 12);
    }
}
