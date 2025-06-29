# NLTK - Cheatsheet for Developers

## Introduction-What-is-NLTK?

> Seaborn is a powerful and popular Python library specifically designed for creating attractive and informative statistical graphics. Built on top of Matplotlib, it offers a higher-level interface that simplifies the process of generating complex visualizations, especially when working with Pandas DataFrames. Seaborn excels at helping users explore and understand their data by providing a wide array of specialized plot types for visualizing relationships between variables, distributions, and categorical data, often incorporating statistical estimations like regression lines or confidence intervals automatically. Its appealing default styles, color palettes, and concise syntax make it a go-to tool for data scientists and analysts seeking to effectively communicate insights from their data with minimal coding effort.


## 1. Setup & Data Acquisition

> This section covers the initial steps of bringing the Seaborn library into your Python script and loading its convenient built-in datasets for quick exploration.

|Command | description|
|----------|-------------|
|`pip install nltk`|	Installs the NLTK library using pip.|
|`import nltk`|	Imports the NLTK library into your Python script/notebook. This is the standard way to begin using NLTK functionalities.|
|`nltk.download()`|Opens the NLTK Downloader GUI, allowing you to selectively download corpora, models, and other NLTK data. This is the most common way to get necessary data for various NLTK tasks.|
|`nltk.download('all')`|Downloads all available NLTK corpora, models, and data. Caution: This is a very large download and might take a significant amount of time and disk space.|
|`nltk.download('punkt')`|Downloads the 'punkt' tokenizer models, essential for word and sentence tokenization. This is a very frequently used download.|
|`nltk.download('stopwords')`|Downloads the standard list of stopwords for various languages, used for text cleaning.|
|`nltk.download('wordnet')`|Downloads the WordNet lexicon, which is crucial for lemmatization and semantic analysis.|
|`nltk.download('averaged_perceptron_tagger')`|Downloads the pre-trained model for Part-of-Speech (POS) tagging.|
|`nltk.download('maxent_ne_chunker')`|Downloads the pre-trained maximum entropy model used for Named Entity Recognition (NER).|
|`nltk.download('words')`|Downloads a corpus of common English words, often used for basic spell-checking or word validation.|
|`nltk.data.path`|Displays the list of directories where NLTK looks for its data files. You can also append custom paths here.|
|`nltk.data.path.append('/path/to/your/nltk_data')`|Adds a custom directory to NLTK's search path for data. Useful if you store data in a non-standard location.|


## 2. Basic Text Preprocessing

> This section covers the initial steps of bringing the Seaborn library into your Python script and loading its convenient built-in datasets for quick exploration.

|Command | description|
|----------|-------------|
|`from nltk.tokenize import word_tokenize <br> word_tokenize(text)`|Tokenizes a given text string into individual words and punctuation marks. It handles contractions and common linguistic nuances.|
|`from nltk.tokenize import sent_tokenize <br> sent_tokenize(text)`|Tokenizes a given text string into a list of sentences. It uses an unsupervised algorithm (Punkt tokenizer) that can be trained on a specific language.|
|`text.lower()`|Converts all characters in a given text string to lowercase. This is crucial for normalizing text and ensuring that "Word" and "word" are treated as the same token.|
|`text.upper()`|Converts all characters in a given text string to uppercase. Less commonly used for preprocessing than lowercasing, but useful in specific scenarios.|
|`import string <br> text.translate(str.maketrans('', '', string.punctuation))`|Removes all standard punctuation characters from a given text string. string.punctuation is a string of common punctuation marks.|
|`import re <br> re.sub(r'[^\w\s]', '', text)`|Removes all non-alphanumeric characters (i.e., punctuation and symbols) from a given text string using regular expressions. \w matches word characters (alphanumeric + underscore), and \s matches whitespace characters.|
|`re.sub(r'\d+', '', text)`|Removes all digit characters from a given text string using regular expressions. Useful if numbers are not relevant to your analysis.|
|`re.sub(r'\s+', ' ', text).strip()`|Replaces multiple whitespace characters (spaces, tabs, newlines) with a single space and then removes leading/trailing whitespace. Helps in standardizing whitespace.|
|`from nltk.tokenize import RegexpTokenizer <br> tokenizer = RegexpTokenizer(r'\w+') <br> tokenizer.tokenize(text)`|Tokenizes a given text string based on a regular expression pattern. The example r'\w+' extracts only word characters, effectively removing punctuation and numbers (depending on your definition of 'word').|
|`from nltk.tokenize.casual import TweetTokenizer <br> tokenizer = TweetTokenizer() <br> tokenizer.tokenize(tweet_text)`|A specialized tokenizer from NLTK designed for noisy text like tweets. It handles emoticons, hashtags, and user mentions more gracefully than general tokenizers.|


## 3. Text Normalization

> This section covers the initial steps of bringing the Seaborn library into your Python script and loading its convenient built-in datasets for quick exploration.

|Command | description|
|----------|-------------|
|`from nltk.corpus import stopwords`|Imports the stopwords corpus from NLTK. This corpus contains common words that often carry little significant meaning in text analysis (e.g., "the", "is", "and").|
|`set(stopwords.words('english'))`|Loads the list of English stopwords as a set for efficient lookup. Replace 'english' with other languages if needed (e.g., 'spanish', 'french').|
|`[word for word in tokens if word.lower() not in stop_words]`|Example Python list comprehension to filter out stopwords from a list of tokens. It converts words to lowercase before checking against the stop_words set.|
|`from nltk.stem import PorterStemmer`|Imports the PorterStemmer class. The Porter Stemmer is a widely used algorithm for removing morphological affixes from words.|
|`ps = PorterStemmer()`|Initializes an instance of the PorterStemmer.|
|`ps.stem(word)`|Stems a single word using the Porter Stemmer. For example, ps.stem('running') returns 'run'.|
|`from nltk.stem import SnowballStemmer`|Imports the SnowballStemmer class. The Snowball Stemmer (also known as the "Porter2" stemmer) is an improved version of the Porter Stemmer, often providing better results and supporting multiple languages.|
|`ss = SnowballStemmer('english')`|Initializes an instance of the SnowballStemmer for a specific language (e.g., 'english'). Other languages include 'french', 'german', 'spanish', etc.|
|`ss.stem(word)`|Stems a single word using the Snowball Stemmer. For example, ss.stem('generously') returns 'generous'.|
|`from nltk.stem import LancasterStemmer`|Imports the LancasterStemmer class. The Lancaster Stemmer is a more aggressive stemmer, often producing shorter stems than Porter or Snowball.|
|`ls = LancasterStemmer()`|Initializes an instance of the LancasterStemmer.|
|`ls.stem(word)`|Stems a single word using the Lancaster Stemmer. For example, ls.stem('maximum') returns 'maxim'.|
|`from nltk.stem import WordNetLemmatizer`|Imports the WordNetLemmatizer class. Lemmatization aims to reduce words to their base or dictionary form (lemma), taking into account the word's part-of-speech.|
|`lemmatizer = WordNetLemmatizer()`|Initializes an instance of the WordNetLemmatizer. For effective lemmatization, you often need to download the WordNet corpus (e.g., nltk.download('wordnet')).|
|`lemmatizer.lemmatize(word)`|Lemmatizes a single word to its base form. By default, it assumes the word is a noun. Example: lemmatizer.lemmatize('rocks') returns 'rock', but lemmatizer.lemmatize('running') returns 'running' (as it assumes noun).|
|`lemmatizer.lemmatize(word, pos='v')`|Lemmatizes a single word by specifying its Part-of-Speech (POS). Common POS tags: 'n' (noun, default), 'v' (verb), 'a' (adjective), 'r' (adverb). Example: lemmatizer.lemmatize('running', pos='v') returns 'run'.|
|`lemmatizer.lemmatize(word, pos='a')`|Lemmatizes a word as an adjective. Example: lemmatizer.lemmatize('better', pos='a') returns 'good'.|


## 4. Lexical Analysis & Frequency

> This section covers the initial steps of bringing the Seaborn library into your Python script and loading its convenient built-in datasets for quick exploration.

|Command | description|
|----------|-------------|
|`text.concordance(word)`|Displays information about occurrences of word in the context of the text object. This shows the word with its left and right context.|
|`text.dispersion_plot([word1, word2, ...])`|Plots the occurrences of the specified words in the text object, showing how frequently and where they appear across the text. Requires Matplotlib to be installed.|
|`text.collocations()`|Finds words that frequently appear together (collocations) in the text object. By default, it finds bigrams.|
|`text.findall(regexp)`|Finds all occurrences of the regular expression regexp in the text object.|
|`FreqDist(list_of_words)`|Creates a frequency distribution object from a list of words. This counts the occurrences of each unique word.|
|`fdist[word]`|Accesses the frequency count of a specific word in the fdist (FreqDist) object.|
|`fdist.most_common(n)`|Returns a list of the n most common words and their frequencies from the fdist object.|
|`fdist.hapaxes()`|Returns a list of words that appear only once (hapaxes) in the fdist object.|
|`fdist.keys()`|Returns a list of all unique words (keys) in the fdist object.|
|`fdist.values()`|Returns a list of the frequency counts for all unique words in the fdist object.|
|`fdist.items()`|Returns a list of (word, frequency) pairs for all unique words in the fdist object.|
|`fdist.plot(n, cumulative=False)`|Plots the n most common words in the fdist object. cumulative=True plots the cumulative frequency. Requires Matplotlib.|
|`fdist.tabulate(n)`|Prints a table of the n most common words and their frequencies from the fdist object to the console.|
|`fdist.max()`|Returns the word with the highest frequency in the fdist object.|
|`fdist.N()`|Returns the total number of items (words, including repetitions) in the fdist object. Equivalent to len(list_of_words).|
|`fdist.B()`|Returns the total number of unique items (words) in the fdist object. Equivalent to len(set(list_of_words)).|
|`fdist.freq(word)`|Returns the frequency of a word as a fraction of the total number of words in the fdist object.|
|`nltk.bigrams(list_of_words)`|Generates an iterator of bigrams (pairs of adjacent words) from a list of words.|
|`nltk.trigrams(list_of_words)`|Generates an iterator of trigrams (sequences of three adjacent words) from a list of words.|
|`nltk.ngrams(list_of_words, n)`|Generates an iterator of n-grams (sequences of n adjacent words) from a list of words.|


## 5. Part-of-Speech (POS) Tagging

> This section covers the initial steps of bringing the Seaborn library into your Python script and loading its convenient built-in datasets for quick exploration.

|Command | description|
|----------|-------------|
|`nltk.pos_tag(tokens)`|Tags each word in a list of tokens with its corresponding Part-of-Speech (POS) tag. Returns a list of (word, tag) tuples. Uses the default Treebank tagger.|
|`nltk.tag.pos_tag(tokens, tagset=None, lang='eng')`|More explicit version of nltk.pos_tag. tagset can be specified (e.g., 'universal' for Universal POS Tagset, 'brown' for Brown corpus tags). lang can be used to specify language (e.g., 'eng' for English).|
|`nltk.word_tokenize(text)`|Tokenizes a text string into words. This is often the first step before POS tagging, as pos_tag expects a list of words.|
|`nltk.data.load('taggers/averaged_perceptron_tagger/english.pickle')`|Loads a pre-trained English POS tagger. This is the tagger typically used by nltk.pos_tag by default. Useful if you want to use the tagger object directly for more control or advanced scenarios.|
|`tagger.tag(tokens)`|If you've loaded a tagger object (e.g., using nltk.data.load()), you can use its .tag() method to perform POS tagging on a list of tokens.|
|`nltk.tag.untag(tagged_words)`|Removes the POS tags from a list of (word, tag) tuples, returning just the list of words.|
|`nltk.help.upenn_tagset()`|Displays a help message providing descriptions for the standard Penn Treebank POS tags (e.g., NN, VBZ, JJ). This is crucial for understanding what the tags mean.|
|`nltk.help.upenn_tagset('NN')`|Displays a detailed description for a specific Penn Treebank tag, e.g., 'NN' for noun, singular.|
|`nltk.tag.str2tuple('word/TAG')`|Converts a string like 'word/TAG' into a (word, TAG) tuple. Useful when you have pre-tagged text in this format.|
|`nltk.tag.tuple2str(('word', 'TAG'))`|Converts a (word, TAG) tuple back into a string like 'word/TAG'.|
|`nltk.corpus.brown.tagged_words()`|Accesses words from the Brown Corpus that are already pre-tagged with POS information. Useful for studying tagged data or training/evaluating taggers.|
|`nltk.corpus.brown.tagged_sents()`|Accesses sentences from the Brown Corpus, where each sentence is a list of (word, tag) tuples.|
|`nltk.corpus.treebank.tagged_words()`|Accesses words from the Penn Treebank Corpus that are pre-tagged.|
|`nltk.corpus.treebank.tagged_sents()`|Accesses sentences from the Penn Treebank Corpus with pre-tagged words.|
|`nltk.tag.AccuracyTagger(reference_tags)`|A basic tagger that returns tags based on a reference list. Primarily for demonstration or simple testing, not for robust tagging.|
|`nltk.tag.UnigramTagger(train_sents)`|A simple POS tagger that assigns the most frequent tag to each word based on training data. train_sents should be a list of tagged sentences.|
|`nltk.tag.BigramTagger(train_sents)`|A POS tagger that considers the previous word's tag when assigning a tag, in addition to the word itself. Requires training data.|
|`nltk.tag.TrigramTagger(train_sents)`|Similar to BigramTagger but considers the tags of the two previous words. Requires more training data to be effective.|
|`nltk.tag.RegexpTagger([('pattern', 'TAG')])`|A rule-based tagger that uses regular expressions to assign tags. Rules are applied sequentially. E.g., r'.*ing$' for VBG.|
|`nltk.tag.DefaultTagger('NN')`|A very simple tagger that assigns the same default tag (e.g., 'NN' for noun) to every word. Often used as a fallback in a backoff chain.|
|`tagger.evaluate(test_sents)`|Evaluates the accuracy of a trained tagger object against a set of test_sents (which should be tagged sentences). Returns a float representing accuracy.|
|`nltk.tag.sequential.PerceptronTagger(load=True)`|Initializes the PerceptronTagger. This is a robust and common tagger for English, capable of learning from data. load=True loads the default pre-trained model.|
|`tagger.train(training_sents)`|Trains a PerceptronTagger (or other trainable tagger) on a list of training_sents (tagged sentences).|
|`nltk.tag.map_tag(tagset_source, tagset_target, tag)`|Maps a tag from one tagset (e.g., Penn Treebank) to another (e.g., Universal POS Tagset). tagset_source and tagset_target can be 'universal', 'brown', 'wsj', etc.|
|`nltk.corpus.treebank.tagset()`|Returns the set of all tags used in the Penn Treebank corpus.|
|`nltk.corpus.brown.tagset()`|Returns the set of all tags used in the Brown corpus.|


## 6. Named Entity Recognition (NER) & Chunking

> This section covers the initial steps of bringing the Seaborn library into your Python script and loading its convenient built-in datasets for quick exploration.

|Command | description|
|----------|-------------|
|`nltk.chunk.ne_chunk(tagged_words, binary=False)`|Identifies named entities in a list of POS-tagged words. Returns a tree structure where named entities are grouped. If binary=True, it only distinguishes between named entities and non-named entities (e.g., NE vs. None). If binary=False, it attempts to classify the type of named entity (e.g., PERSON, ORGANIZATION, GPE). Requires the maxent_ne_chunker and words corpora.|
|`nltk.chunk.RegexpParser(grammar)`|Creates a regular expression parser that can be used for chunking. The grammar is a string defining chunking rules using regular expressions.|
|`chunker.parse(tagged_words)`|Parses a list of POS-tagged words using a RegexpParser object (chunker). Returns a tree structure where chunks are identified according to the grammar rules.|
|`tree.draw()`|Opens a graphical window to display the parse tree (or chunk tree) generated by ne_chunk or RegexpParser. Useful for visualizing the identified chunks.|
|`nltk.chunk.conll2000.chunked_sents('train.txt', chunk_types=['NP'])`|Loads chunked sentences from the CoNLL-2000 chunking corpus, which is pre-annotated with noun phrase (NP), verb phrase (VP), and prepositional phrase (PP) chunks. Useful for training and evaluating chunkers. You can specify chunk_types.|
|`nltk.chunk.maxent.MaxentChunker`|A maxent-based chunker that can be trained on chunked corpora (like CoNLL-2000). Provides methods for training and tagging.|
|`MaxentChunker.train(training_sents)`|Trains a MaxentChunker model on a list of chunked sentences (e.g., from conll2000.chunked_sents()).|
|`MaxentChunker.tag(tagged_words)`|Tags a list of POS-tagged words with chunk labels (e.g., IOB format: B-NP, I-NP, O).|
|`nltk.chunk.ChunkScore()`|An object used to evaluate the performance of a chunker. It can calculate precision, recall, and F-measure for identified chunks.|
|`chunkscorer.score(gold_sentences, test_sentences)`|Calculates the chunking performance by comparing a list of "gold standard" chunked sentences with a list of "test" chunked sentences (produced by your chunker).|
|`nltk.tag.stanford.StanfordNERTagger(model_path, jar_path, encoding='utf8')`|(Requires Stanford NER installation) Creates an interface to the Stanford Named Entity Recognizer. Useful for more robust and accurate NER, especially for various entity types. model_path is the path to the Stanford NER model, jar_path is the path to the Stanford NER JAR file.|
|`stanford_ner_tagger.tag(tokens)`|(Used with StanfordNERTagger) Tags a list of tokens with named entity labels (e.g., (word, 'O'), (word, 'PERSON')).|


## 7. Syntactic Parsing

> This section covers the initial steps of bringing the Seaborn library into your Python script and loading its convenient built-in datasets for quick exploration.

|Command | description|
|----------|-------------|
|`nltk.CFG.fromstring(grammar_string)`|Creates a Context-Free Grammar (CFG) object from a string representation. The string defines the grammar rules (productions).|
|`grammar.productions()`|Returns a list of all production rules defined in the grammar object.|
|`grammar.start()`|Returns the start symbol of the grammar.|
|`nltk.parse.RecursiveDescentParser(grammar)`|Creates a recursive descent parser object for the given grammar. Suitable for simple grammars.|
|`nltk.parse.ShiftReduceParser(grammar)`|Creates a shift-reduce parser object for the given grammar. Offers a more efficient parsing strategy than recursive descent.|
|`nltk.parse.ChartParser(grammar)`|Creates a chart parser (e.g., Cocke-Younger-Kasami (CYK) parser) object for the given `grammar). More robust for ambiguous grammars.|
|`parser.parse(word_list)`|Attempts to parse a list of words (word_list) using the initialized parser. Returns an iterator over possible parse trees.|
|`nltk.parse.ViterbiParser(grammar)`|Creates a Viterbi parser. This parser is designed for probabilistic context-free grammars (PCFG) and finds the most probable parse tree. (Requires nltk.PCFG.fromstring for grammar).|
|`nltk.Tree.fromstring(tree_string)`|Creates a Tree object from a standard bracketed tree string representation (e.g., "(S (NP John) (VP (V ate) (NP pizza)))").|
|`tree.pretty_print()`|Prints a visually appealing, indented representation of the tree object to the console.|
|`tree.draw()`|Opens a graphical window to display the tree object (requires Matplotlib and optionally Ghostscript for saving).|
|`tree.label()`|Returns the label (non-terminal symbol) of the current node in the tree.|
|`tree[i]`|Accesses the i-th child of the tree node. Children can be subtrees or leaf nodes (words).|
|`tree.leaves()`|Returns a list of all leaf nodes (terminal symbols/words) in the tree.|
|`tree.height()`|Returns the height of the tree.|
|`tree.flatten()`|Returns a new tree where all non-terminal nodes have been removed, leaving only the leaves under a single root.|
|`tree.chomsky_normal_form()`|Transforms the tree into Chomsky Normal Form (CNF) in-place. Useful for certain parsing algorithms.|
|`tree.collapse_unary(join_with_dash=True)`|Collapses unary productions (e.g., A -> B) in the tree by combining the labels, useful for simplifying tree structures.|
|`tree.productions()`|Extracts the context-free grammar production rules implied by the tree structure.|
|`tree.subtrees()`|Returns an iterator over all subtrees contained within the tree, including the tree itself.|
|`tree.pos()`|Returns a list of (word, POS-tag) tuples from the leaves of the tree if the leaves are POS-tagged.|
|`parser.parse(tokens, trace=1)`|When parsing, trace=1 (or higher) will show the steps the parser takes during parsing, which is useful for debugging grammars.|
|`nltk.DependencyGraph(conll_string)`|Creates a dependency graph object from a CoNLL formatted string. CoNLL is a common format for representing dependency trees.|
|`dep_graph.tree()`|Converts the DependencyGraph into an nltk.Tree object, often representing the dependency relations as a tree structure (though not all dependency graphs are trees).|
|`dep_graph.nodes`|A dictionary representing the nodes (words) in the dependency graph, including their properties like word, lemma, POS, etc.|
|`dep_graph.root`|Returns the ID of the root node in the dependency graph.|
|`dep_graph.triples()`|Returns a list of (head, relation, dependent) triples representing the dependency relations in the graph.|
|`dep_graph.pretty_print()`|Prints a textual representation of the dependency graph.|
|`nltk.parse.malt.MaltParser(tagger=None, parser_path=None, working_dir=None, dynet_mem='200m')`|An interface to the external MaltParser (requires MaltParser JAR file and Java installed). NLTK itself does not provide a built-in robust dependency parser. This command sets up the interface to use an external tool.|
|`malt_parser.parse_sents(sentence_list, verbose=False, root_node='ROOT')`|Parses a list of sentences using the configured MaltParser. Returns a list of DependencyGraph objects.|
|`nltk.parse.corenlp.CoreNLPParser(url='http://localhost:9000')`|An interface to Stanford CoreNLP's parser (requires a running CoreNLP server). Similar to MaltParser, NLTK integrates with external, more powerful dependency parsers.|
|`corenlp_parser.parse_sents(sentence_list, properties={'annotators': 'parse'}, verbose=False)`|Parses a list of sentences using the configured CoreNLP parser. Returns an iterator over parse trees (including dependency trees if requested in properties).|


## 8. Semantic Analysis (WordNet)

> This section covers the initial steps of bringing the Seaborn library into your Python script and loading its convenient built-in datasets for quick exploration.

|Command | description|
|----------|-------------|
|`from nltk.corpus import wordnet as wn`|Imports the WordNet corpus with a commonly used alias wn. This is the essential first step to interact with WordNet.
|`wn.synsets('word')`|Returns a list of all Synsets (sets of cognitive synonyms) that contain the specified 'word'. Each Synset represents a distinct concept. E.g., wn.synsets('dog') will give synsets for the animal, the bad person, etc.|
|`wn.synset('word.pos.nn')`|Retrieves a specific Synset using its unique name. The format is word.pos.nn, where word is the lemma, pos is the part of speech (e.g., 'n' for noun, 'v' for verb, 'a' for adjective, 'r' for adverb), and nn is a two-digit sense number. E.g., wn.synset('dog.n.01') refers to the first sense of 'dog' as a noun (the animal).|
|`synset.lemmas()`|Returns a list of Lemma objects associated with a given Synset. Each lemma represents a specific word form in that sense.|
|`synset.definition()`|Returns the textual definition of the concept represented by the Synset.|
|`synset.examples()`|Returns a list of example sentences that illustrate the usage of the Synset.|
|`synset.name()`|Returns the full unique name of the Synset (e.g., 'dog.n.01').|
|`synset.pos()`|Returns the part of speech of the Synset (e.g., 'n', 'v', 'a', 'r').|
|`lemma.name()`|Returns the word string of the Lemma.|
|`lemma.synset()`|Returns the Synset that the Lemma belongs to.|
|`lemma.antonyms()`|Returns a list of Lemma objects that are antonyms of the current Lemma. This explicitly finds lexical antonyms.|
|`wn.lemmas('word')`|Returns a list of all Lemma objects for the given 'word' across all its Synsets.|
|`wn.all_synsets('pos')`|Returns an iterator over all Synsets for a given part of speech (e.g., wn.all_synsets('n') for all nouns).|
|`wn.all_lemma_names()`|Returns a list of all unique lemma names (word strings) in WordNet.|
|`wn.langs()`|Returns a list of languages supported by the WordNet instance (e.g., ['eng'] for English).|
|`synset.hypernyms()`|Returns a list of Synsets that are direct hypernyms (more general concepts) of the current Synset. E.g., for dog.n.01, this might return canine.n.02.|
|`synset.hypernym_paths()`|Returns a list of all possible hypernym paths (hierarchical relationships) from the current Synset up to the root.|
|`synset.hyponyms()`|Returns a list of Synsets that are direct hyponyms (more specific concepts) of the current Synset. E.g., for canine.n.02, this might return dog.n.01.|
|`synset.member_holonyms()`|Returns a list of Synsets that are larger wholes of which the current Synset is a member. E.g., for tree.n.01, this might return forest.n.01.|
|`synset.substance_holonyms()`|Returns a list of Synsets that are substances of which the current Synset is a part.|
|`synset.part_holonyms()`|Returns a list of Synsets that are parts of which the current Synset is a component.|
|`synset.member_meronyms()`|Returns a list of Synsets that are members of the current Synset. E.g., for forest.n.01, this might return tree.n.01.|
|`synset.substance_meronyms()`|Returns a list of Synsets that are substances that make up the current Synset.|
|`synset.part_meronyms()`|Returns a list of Synsets that are parts of the current Synset.|
|`synset1.wup_similarity(synset2)`|Calculates the Wu-Palmer Similarity between two Synsets. This measures the depth of the Least Common Subsumer (LCS) in the hierarchy, indicating how similar two concepts are based on their shared ancestors. Returns a score between 0.0 and 1.0.|
|`synset1.path_similarity(synset2)`|Calculates the Path Similarity between two Synsets. This is based on the shortest path that connects the two synsets in the WordNet graph. Returns a score between 0.0 and 1.0, where 1.0 means identical.|
|`synset1.lch_similarity(synset2)`|Calculates the Leacock-Chodorow Similarity between two Synsets. This measures similarity based on the shortest path length between two synsets and the maximum depth of the taxonomy. Returns a score from 0.0 to 1.0.|
|`synset1.res_similarity(synset2, ic)`|Calculates the Resnik Similarity between two Synsets, requiring an Information Content (IC) corpus. Measures similarity based on the IC of their Lowest Common Subsumer (LCS). Requires |`nltk.download('wordnet_ic')`| and loading an IC corpus (e.g., ic = wn.ic('ic-brown.dat')).|
|`synset1.jcn_similarity(synset2, ic)`|Calculates the Jiang-Conrath Similarity between two Synsets, also requiring an Information Content (IC) corpus. Measures similarity based on the sum of the IC of the two synsets minus twice the IC of their LCS.|
|`wn.synsets('word', pos='n')`|Returns Synsets for a specific part of speech. Useful for disambiguation, e.g., getting only noun senses of 'bank'.|
|`wn.morphy('word', pos='n')`|Attempts to find the base form (lemma) of a word, similar to lemmatization, by consulting WordNet's morphological rules. The pos argument helps guide the search.|


## 8. Semantic Analysis (WordNet)

> This section covers the initial steps of bringing the Seaborn library into your Python script and loading its convenient built-in datasets for quick exploration.

|Command | description|
|----------|-------------|
|`from nltk.corpus import wordnet as wn`|Imports the WordNet corpus with a commonly used alias wn. This is the essential first step to interact with WordNet.







