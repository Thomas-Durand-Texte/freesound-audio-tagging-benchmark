
# Freesound Dataset Kaggle 2019 (FSDKaggle2019)



## Citation

If you use the FSDKaggle2019 dataset or part of it, please cite our <a href="https://arxiv.org/abs/1906.02975">DCASE 2019 paper</a>:

>Eduardo Fonseca, Manoj Plakal, Frederic Font, Daniel P. W. Ellis, Xavier Serra. "Audio tagging with noisy labels and minimal supervision". *Proceedings of the DCASE 2019 Workshop* (2019)

You can also consider citing our <a href="https://repositori.upf.edu/bitstream/handle/10230/33299/fonseca_ismir17_freesound.pdf?sequence=1&isAllowed=y">ISMIR 2017 paper</a>, which describes how we gathered the manual annotations included in FSDKaggle2019.

>Eduardo Fonseca, Jordi Pons, Xavier Favory, Frederic Font, Dmitry Bogdanov, Andres Ferraro, Sergio Oramas, Alastair Porter, and Xavier Serra, "Freesound Datasets: A Platform for the Creation of Open Audio Datasets", In *Proceedings of the 18th International Society for Music Information Retrieval Conference*, Suzhou, China, 2017

### Data curators

Eduardo Fonseca, Manoj Plakal, Xavier Favory, Jordi Pons

### Contact

You are welcome to contact Eduardo Fonseca should you have any questions at eduardo.fonseca@upf.edu.

## About FSDKaggle2019

Freesound Dataset Kaggle 2019 (or **FSDKaggle2019** for short) is an audio dataset containing 29,266 audio files annotated with 80 labels of the <a href="https://research.google.com/audioset/ontology/index.html">AudioSet Ontology</a> [1]. FSDKaggle2019 has been used for the Task 2 of the *Detection and Classification of Acoustic Scenes and Events* (DCASE) Challenge 2019. Please visit the <a href="http://dcase.community/challenge2019/task-audio-tagging">DCASE2019 Challenge Task 2 website</a> for more information. This Task was hosted on the Kaggle platform as a competition titled <a href="https://www.kaggle.com/c/freesound-audio-tagging-2019">Freesound Audio Tagging 2019</a>. It was organized by researchers from the Music Technology Group of Universitat Pompeu Fabra, and from the Sound Understanding team at Google AI Perception. The competition intended to provide insight towards the development of broadly-applicable sound event classifiers able to cope with label noise and minimal supervision conditions. 

FSDKaggle2019 employs audio clips from the following sources:

 - Freesound Dataset (<a href="https://annotator.freesound.org/fsd/">FSD</a>): a dataset being collected at the <a href="https://www.upf.edu/web/mtg">MTG-UPF</a> based on <a href="https://freesound.org/">Freesound</a> content organized with the <a href="https://research.google.com/audioset/ontology/index.html">AudioSet Ontology</a>
 - The soundtracks of a pool of Flickr videos taken from the <a href="http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/">Yahoo Flickr Creative Commons 100M dataset (YFCC)</a>

The audio data is labeled using a vocabulary of 80 labels from Google’s <a href="https://research.google.com/audioset/ontology/index.html">AudioSet Ontology</a> [1], covering diverse topics: Guitar and other Musical Instruments, Percussion, Water, Digestive, Respiratory sounds, Human voice, Human locomotion, Hands, Human group actions, Insect, Domestic animals, Glass, Liquid, Motor vehicle (road), Mechanisms, Doors, and a variety of Domestic sounds. The full list of categories can be inspected in `vocabulary.csv` (see [Files & Download](#files-&-download) below). The goal of the task was to build a multi-label audio tagging system that can predict appropriate label(s) for each audio clip in a test set.

What follows is a summary of some of the most relevant characteristics of FSDKaggle2019. Nevertheless, it is **highly recommended** to read our <a href="https://arxiv.org/abs/1906.02975">DCASE 2019 paper</a> for a more in-depth description of the dataset and how it was built.


### Ground Truth Labels

The ground truth labels are provided at the clip-level, and express the presence of a sound category in the audio clip, hence can be considered *weak* labels or tags. Audio clips have variable lengths (roughly from 0.3 to 30s).

The audio content from <a href="https://annotator.freesound.org/fsd/">FSD</a> has been manually labeled by humans following a data labeling process using the <a href="https://annotator.freesound.org/">Freesound Annotator</a> platform. Most labels have inter-annotator agreement but not all of them. More details about the data labeling process and the <a href="https://annotator.freesound.org/">Freesound Annotator</a> can be found in [2].

The <a href="http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/">YFCC</a> soundtracks were labeled using automated heuristics applied to the audio content and metadata of the original Flickr clips. Hence, a substantial amount of label noise can be expected. The label noise can vary widely in amount and type depending on the category, including in- and out-of-vocabulary noises. More information about some of the types of label noise that can be encountered is available in [3].

Specifically, FSDKaggle2019 features **three types of label quality**, one for each set in the dataset:

 - **curated train set**: correct (but potentially incomplete) labels
 - **noisy train set**: noisy labels
 - **test set**: correct and complete labels

Further details can be found below in the sections for each set.

### Format

All audio clips are provided as uncompressed PCM 16 bit, 44.1 kHz, mono audio files.

## Data split

FSDKaggle2019 consists of **two train sets** and **one test set**. The idea is to limit the supervision provided for training (i.e., the manually-labeled, hence reliable, data), thus promoting approaches to deal with label noise.

### Curated train set
The **curated train set** consists of manually-labeled data from <a href="https://annotator.freesound.org/fsd/">FSD</a>. 

 - Number of clips/class: 75 except in a few cases (where there are less)
 - Total number of clips: 4970  
 - Avg number of labels/clip: 1.2  
 - Total duration: 10.5 hours  

The duration of the audio clips ranges from 0.3 to 30s due to the diversity of the sound categories and the preferences of Freesound users when recording/uploading sounds. Labels are correct but potentially incomplete. It can happen that a few of these audio clips present additional acoustic material beyond the provided ground truth label(s).

### Noisy train set
The **noisy train set** is a larger set of noisy web audio data from Flickr videos taken from the <a href="http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/">YFCC</a> dataset [5].

 - Number of clips/class: 300
 - Total number of clips: 19,815
 - Avg number of labels/clip: 1.2
 - Total duration: ~80 hours

The duration of the audio clips ranges from 1s to 15s, with the vast majority lasting 15s. Labels are automatically generated and purposefully noisy. No human validation is involved. The label noise can vary widely in amount and type depending on the category, including in- and out-of-vocabulary noises.

Considering the numbers above, the per-class data distribution available for training is, for most of the classes, 300 clips from the noisy train set and 75 clips from the curated train set. This means 80% noisy / 20% curated at the clip level, while at the duration level the proportion is more extreme considering the variable-length clips.


### Test set

The **test set** is used for system evaluation and consists of manually-labeled data from <a href="https://annotator.freesound.org/fsd/">FSD</a>. 

 - Number of clips/class: between 50 and 150
 - Total number of clips: 4481  
 - Avg number of labels/clip: 1.4  
 - Total duration: 12.9 hours  

The acoustic material present in the test set clips is labeled exhaustively using the aforementioned vocabulary of 80 classes. Most labels have inter-annotator agreement but not all of them. Except human error, the label(s) are correct and complete considering the target vocabulary; nonetheless, a few clips could still present additional (unlabeled) acoustic content out of the vocabulary.

During the <a href="http://dcase.community/challenge2019/task-audio-tagging">DCASE2019 Challenge Task 2</a>, the test set was split into two subsets, for the **public** and **private** leaderboards, and only the data corresponding to the *public* leaderboard was provided. **In this current package you will find the full test set with all the test labels**. To allow comparison with previous work, the file `test_post_competition.csv ` includes a flag to determine the corresponding leaderboard (public or private) for each test clip (see more info in [Files & Download](#files-&-download) below).


### Acoustic mismatch

As mentioned before, FSDKaggle2019 uses audio clips from two sources:

 - <a href="https://annotator.freesound.org/fsd/">FSD</a>: curated train set and test set, and
 - <a href="http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/">YFCC</a>: noisy train set.

While the sources of audio (Freesound and Flickr) are collaboratively contributed and pretty diverse themselves, a certain acoustic mismatch can be expected between <a href="https://annotator.freesound.org/fsd/">FSD</a> and <a href="http://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-100-million-creative-commons-dataset/">YFCC</a>. We conjecture this mismatch comes from a variety of reasons.   
For example, through acoustic inspection of a small sample of both data sources, we find a higher percentage of high quality recordings in FSD. In addition, audio clips in Freesound are typically recorded with the purpose of capturing audio, which is not necessarily the case in YFCC.

This mismatch can have an impact in the evaluation, considering that most of the train data come from YFCC, while all test data are drawn from FSD. This constraint (i.e., noisy training data coming from a different web audio source than the test set) is sometimes a real-world condition.


## License

All clips in FSDKaggle2019 are released under Creative Commons (CC) licenses. For attribution purposes and to facilitate attribution of these files to third parties, we include a mapping from the audio clips to their corresponding licenses.

- **Curated train set and test set**. All clips in **Freesound** are released under Creative Commons (CC) licenses, and each audio clip has its own license as defined by the audio clip uploader in Freesound, some of them requiring attribution to their original authors and some forbidding further commercial reuse.  The licenses are specified in the files `train_curated_post_competition.csv` and `test_post_competition.csv`. These licenses are CC0, CC-BY, CC-BY-NC and CC Sampling+.

- **Noisy train set**. Similarly, the licenses of the soundtracks from **Flickr** used in FSDKaggle2019 are specified in the file `train_noisy_post_competition.csv`. These licenses are CC-BY and CC BY-SA.

In addition, FSDKaggle2019 as a whole is the result of a curation process and it has an additional license. FSDKaggle2019 is released under <a href="https://creativecommons.org/licenses/by/4.0/">CC-BY</a>. This license is specified in the `LICENSE-DATASET` file downloaded with the `FSDKaggle2019.doc` zip file.

 
## Files & Download

FSDKaggle2019 can be downloaded as a series of zip files with the following directory structure:

<div class="highlight"><pre><span></span>root
│  
└───FSDKaggle2019.audio_train_curated/               Audio clips in the curated train set
│
└───FSDKaggle2019.audio_train_noisy/                 Audio clips in the noisy train set
│   
└───FSDKaggle2019.audio_test/                        Audio clips in the full test set
│   
└───FSDKaggle2019.meta/                              Files for evaluation setup
│   │            
│   └─── train_curated_post_competition.csv          Ground truth for the curated train set
│   │            
│   └─── train_noisy_post_competition.csv            Ground truth for the noisy train set
│   │            
│   └─── test_post_competition.csv                   Ground truth for the full test set
│   │            
│   └─── vocabulary.csv                              List of sound classes in FSDKaggle2019    
│   
└───FSDKaggle2019.doc/
    │            
    └───README.md                                    The dataset description file that you are reading
    │            
    └───LICENSE-DATASET                              License of the FSDKaggle2019 dataset as an entity   
</pre></div>

**IMPORTANT NOTE:** the original `train_curated.csv` and `train_noisy.csv` files provided during the competition have been updated with more metadata (licenses, Freesound/Flickr ids, etc.) into `train_curated_post_competition.csv` and `train_noisy_post_competition.csv`. Likewise, the original `test.csv` that was not public during the competition is now available with ground truth and metadata as `test_post_competition.csv`.

Each row (i.e. audio clip) of the `train_curated_post_competition.csv` or `train_noisy_post_competition.csv` files contains the following information:

- `fname`: the file name, e.g., `0006ae4e.wav`
- `labels`: the audio classification label(s) (ground truth). Note that the number of labels per clip can be one, eg, `Bark` or more, eg, `"Walk_and_footsteps,Slam"`.
- `freesound_id` or `flickr_video_URL`: the Freesound id or Flickr id for the audio clip
- `license`: the license for the audio clip

Each row (i.e. audio clip) of the `test_post_competition.csv` file contains the following information:

- `fname`: the file name
- `labels`: the audio classification label(s) (ground truth). Note that the number of labels per clip can be one, eg, `Bark` or more, eg, `"Walk_and_footsteps,Slam"`.
- `usage`: string that indicates to which Kaggle leaderboard the clip was associated during the competition: `Public` or `Private`
- `freesound_id`: the Freesound id for the audio clip
- `license`: the license for the audio clip

### Detected corrupted files in the curated train set

The following 5 audio files in the **curated train set** have a wrong label, due to a bug in the file renaming process: `f76181c4.wav`, `77b925c2.wav`, `6a1f682a.wav`, `c7db12aa.wav`, `7752cc8a.wav`.

The audio file `1d44b0bd.wav` in the **curated train set** was found to be corrupted (contains no signal) due to an error in format conversion.

If you find more corrupted files in FSDKaggle2019, please send an email to eduardo.fonseca@upf.edu.

### Download

Each of the folders in the directory structure above is compressed into one corresponding zip file that you can download and unzip with your favorite compression tool. There is one exception: due to the large size of `FSDKaggle2019.audio_train_noisy/`, it is split into 7 files (note the last file is not `*.z07`, but `*.zip`): 

```
FSDKaggle2019.audio_train_noisy.z01
FSDKaggle2019.audio_train_noisy.z02
FSDKaggle2019.audio_train_noisy.z03
FSDKaggle2019.audio_train_noisy.z04
FSDKaggle2019.audio_train_noisy.z05
FSDKaggle2019.audio_train_noisy.z06
FSDKaggle2019.audio_train_noisy.zip
```

In this case, you first have to download the 7 files. Once downloaded, we convert the split archive to a single-file archive. In other words, we merge the 7 files into one zip file called e.g. `unsplit.zip` in your local machine.

```
zip -s 0 FSDKaggle2019.audio_train_noisy.zip --out unsplit.zip
```

Finally, this merged file is unzipped.

```
unzip unsplit.zip
```

## Baseline System

A CNN baseline system for FSDKaggle2019 is available at <a href="https://github.com/DCASE-REPO/dcase2019_task2_baseline">https://github.com/DCASE-REPO/dcase2019_task2_baseline</a>.


## References and links

[1] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. "Audio set: An ontology and human-labeled dataset for audio events." In Proceedings of the International Conference on Acoustics, Speech and Signal Processing, 2017. [<a href="https://ai.google/research/pubs/pub45857">PDF</a>]

[2] Eduardo Fonseca, Jordi Pons, Xavier Favory, Frederic Font, Dmitry Bogdanov, Andres Ferraro, Sergio Oramas, Alastair Porter, and Xavier Serra. "Freesound Datasets: A Platform for the Creation of Open Audio Datasets." In Proceedings of the International Conference on Music Information Retrieval, 2017. [<a href="https://repositori.upf.edu/bitstream/handle/10230/33299/fonseca_ismir17_freesound.pdf">PDF</a>]

[3] Eduardo Fonseca, Manoj Plakal, Daniel P. W. Ellis, Frederic Font, Xavier Favory, and Xavier Serra. "Learning Sound Event Classifiers from Web Audio with Noisy Labels." In Proceedings of the International Conference on Acoustics, Speech and Signal Processing, 2019. [<a href="https://arxiv.org/abs/1901.01189">PDF</a>]

[4] Frederic Font, Gerard Roma, and Xavier Serra. "Freesound technical demo." Proceedings of the 21st ACM international conference on Multimedia, 2013. <a href="https://freesound.org/">https://freesound.org</a>

[5] Bart Thomee, David A. Shamma, Gerald Friedland, Benjamin Elizalde,  Karl Ni,  Douglas Poland,  Damian Borth, and Li-Jia Li, YFCC100M: The New Data in Multimedia Research, Commun. ACM, 59(2):64–73, January 2016

Freesound Annotator: <a href="https://annotator.freesound.org/">https://annotator.freesound.org/</a>  
Freesound: <a href="https://freesound.org">https://freesound.org</a>  
Eduardo Fonseca's personal website: <a href="http://www.eduardofonseca.net/">http://www.eduardofonseca.net/</a>  
More datasets collected by us: <a href="http://www.eduardofonseca.net/datasets/">http://www.eduardofonseca.net/datasets/</a>

## Acknowledgments

This work is partially supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No 688382 <a href="https://www.audiocommons.org/">AudioCommons</a>. Eduardo Fonseca is also sponsored by a <a href="https://ai.googleblog.com/2019/03/google-faculty-research-awards-2018.html">Google Faculty Research Award 2018</a>. We thank everyone who contributed to FSDKaggle2019 with annotations.
