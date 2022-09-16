# e-Braille Tales
This braille OCR application can convert JPEG braille text images into RTF documents, while removing typos for you!

![Image RTF basic mode](https://github.com/LPBeaulieu/e-Braille-Tales/blob/main/e-Braille%20Tales%20Thumbnail.png)
<h3 align="center">e-Braille Tales</h3>
<div align="center">
  
  [![License: AGPL-3.0](https://img.shields.io/badge/License-AGPLv3.0-brightgreen.svg)](https://github.com/LPBeaulieu/TintypeText/blob/main/LICENSE)
  [![GitHub last commit](https://img.shields.io/github/last-commit/LPBeaulieu/e-Braille-Tales)](https://github.com/LPBeaulieu/e-Braille-Tales)
  [![GitHub issues](https://img.shields.io/github/issues/LPBeaulieu/e-Braille-Tales)](https://github.com/LPBeaulieu/e-Braille-Tales)
  
</div>

---

<p align="left"> <b>e-Braille Tales</b> is a tool enabling you to convert scanned braille pages (in JPEG image format and typed on a Perkins Brailler) into Portable Embosser Format (PEF) digitized braille and rich text format (RTF) documents, complete with formatting elements such as alignment, paragraphs, <u>underline</u>, <i>italics</i>, <b>bold</b> and <del>strikethrough</del>, basically allowing you to include any formatting that RTF commands or braille formatting indicators will enable you to do.</p>
<p align="left"> A neat functionality of <b>e-Braille Tales</b> is that the typos (sequence of at least two successive full braille cells)
  automatically get filtered out, and do not appear in the final RTF text nor in the PEF file. The PEF file can in turn be used to print out copies of your work on a braille embosser, or to read them electronically using a refreshable braille display.
  
  - You can get my <b>deep learning model</b> for the Perkins Brailler on my Google Drive (https://drive.google.com/drive/folders/1DUKqYf7wIkRAobC8fYPjum5gFOJqJurv?usp=sharing), where the dataset and other useful information to build your own dataset (if needed) may be found. 
- The code showcased in this github page is the one that was used to generate a model with <b>99.97% optical character recognition (OCR) accuracy</b> with the Perkins Brailler (I'm not affiliated with them, no worries).
  
    <br> 
</p>

## 📝 Table of Contents
- [Dependencies / Limitations](#limitations)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## ⛓️ Dependencies / Limitations <a name = "limitations"></a>
- This Python project relies on the Fastai deep learning library (https://docs.fast.ai/) to generate a convoluted neural network 
  deep learning model, which allows for braille optical character recognition (OCR). It also needs OpenCV to perform image segmentation 
  (to crop the individual characters in the braille page images).
  
- When typing text on the Perkins Brailler, unless a space is included at the end of a line or at the beginning of the next line, the last word on the     line will be merged with the first characters on the next one, up to the next space. As such, the <b>"line continuation without space" braille symbol     ("⠐") is not required and should be avoided</b>, as it could be confused with other braille characters, such as initial-letter contractions. However,     line   continuations with a space ("⠐⠐") can be used without problem in this application.

- In this application a space needs to be included after any RTF command (even though the RTF specifications state that it is an optional space). The       reason for this is that when the code is transposing the braille into printed English, it often needs to determine if the following braille characters   stand alone. A braille character that stands alone means that it is flanked by characters such as empty braille cells ("⠀") or dashes, but not by a       braille character mapping to a letter or number, such that can be found at the end of every RTF command. In other words, <b>you must include a space     after any RTF commands</b>. Here is an example: "This requirement \strike strikes \strike0 me as being important!", which in braille would be written     as follows: "⠠⠹⠀⠗⠑⠟⠥⠊⠗⠑⠰⠞⠀⠸⠡⠎⠞⠗⠊⠅⠑⠀⠎⠞⠗⠊⠅⠑⠎⠀⠸⠡⠎⠞⠗⠊⠅⠑⠼⠚⠀⠍⠑⠀⠵⠀⠆⠬⠀⠊⠍⠏⠕⠗⠞⠁⠝⠞⠖".

- Importantly, <b>the pages must be scanned with the left margin placed on the flatbed scanner in such a way that the shadows produced by the 
  scanner light will face away from the left margin</b> (the shadows will face the right margin of the page, then the page is viewed in landscape mode). 
  This is because the non-white pixels actually result from the presence of shadows, the orientation of which plays a major role in image segmentation     (determining the x and y coordinates of the individual characters) and optical character recognition (OCR). For best results, the braille document 
  should be <b>typed on white braille paper or cardstock and scanned as grayscale images on a flatbed scanner at a 300 dpi resolution with the paper size   setting of the scanner set to letter 8 1/2" x 11" (A4)</b>. The darkness settings of the scanner might also need to be adjusted to acheive an optimal     braille shadow to noise ratio.
  
- <b>The left margin on the Perkins Brailler should be set at its minimal setting</b> to maximize the printable space on the page and to always provide     the same   reference point to the code for the segmentation step. <b>The pixel "x_min" at which the code starts cropping characters on every line needs   to be entered manually in the code, as you initially calibrate the code to your own brailler and scanner combination</b>. In my case, the value of the   variable "x_min" is set to 282       pixels in line 140 of the Python code "e-braille-tales.py". After running the code on a scanned braille text image   of yours, you could then open the     JPEG image overlayed with green character rectangles (see Figure 1 below) in a photo editing software such as       GIMP, in order to locate the pixel value   along the x axis (in landscape mode) at which the segmentation shoud start in each line. 

 - Every brailled line should have braille characters that when taken together contain at least three dots per braille cell row in order to be properly      detected. Should a line only contain characters that do not have dots in one or more of the three braille cell rows, you could make up for the            missing dots by using at least two successive full braille cells ("⠿") before or after the text, which will be interpreted by the code as a typo, and    will not impact the meaningful text on the line in the final Rich Text Format (RTF) and Portable Embosser Format (PEF) files.


## 🏁 Getting Started <a name = "getting_started"></a>

The following instructions will be provided in great detail, as they are intended for a broad audience and will allow to run a copy of <b>e-Braille Tales</b> on a local computer. As the steps 1 through 8 described below are the same as for my other project <b>Tintype Text</b>  (https://github.com/LPBeaulieu/Typewriter-OCR-TintypeText), a link is provided here to an instructional video explaining how to setup Tintype text: https://www.youtube.com/watch?v=FG9WUW6q3dI&list=PL8fAaOg_mhoEZkbQuRgs8MN-QSygAjdil&index=2.

The paths included in the code are formatted for Unix (Linux) operating systems (OS), so the following instructions 
are for Linux OS environments.

<b>Step 1</b>- Go to the command line in your working folder and install the <b>Atom</b> text editor to make editing the code easier:
```
sudo snap install atom --classic
```

<b>Step 2</b>- Create a virtual environment (called <i>env</i>) in your working folder:
```
python3 -m venv env
```

<b>Step 3</b>- Activate the <i>env</i> virtual environment <b>(you will need to do this step every time you use the Python code files)</b> 
in your working folder:
```
source env/bin/activate
```

<b>Step 4</b>- Install <b>PyTorch</b> (Required Fastai library to convert images into a format usable for deep learning) using the following command (or the equivalent command found at https://pytorch.org/get-started/locally/ suitable to your system):
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

<b>Step 5</b>- Install the <i>CPU-only</i> version of <b>Fastai</b> (Deep Learning Python library, the CPU-only version suffices for this application, as the character images are very small in size):
```
pip install fastai
```

<b>Step 6</b>- Install <b>OpenCV</b> (Python library for image segmentation):
```
pip install opencv-python
```

<b>Step 7</b>- Install <b>alive-Progress</b> (Python module for progress bar displayed in command line):
```
pip install alive-progress
```

<b>Step 8</b>- Create the folders "OCR Raw Data" and "Training&Validation Data" in your working folder:
```
mkdir "OCR Raw Data" "Training&Validation Data" 
```
<b>Step 9</b>- You're now ready to use <b>e-Braille Tales</b>! 🎉

## 🎈 Usage <a name="usage"></a>
The "e-braille-tales.py" Python code converts JPEG braille text scans into printed English in the form of an RTF file and digitized braille as a PEF file. You can find instructions on how to use it on my YouTube channel: **LINK**.<br><br>

- In order to submit a scanned braille text page to the code, you need to <b>place the JPEG image in the "OCR Raw Data" subfolder of your working           folder</b>, which you created at step 8 of the "Getting Started" section.

- Then, run the "e-braille-tales.py" Python script by opening the command line from your working folder, such that you will already be in the correct       path and copy and paste the following in command line:  
```
python3 e-braille-tales.py
```

- The first thing that the code will do is perform segmentation (determine the x and y coordinates of every braille character). The segmentation results   are visible in the "Page image files with rectangles" folder, which is created automatically by the code. You might need to <b>adjust the value of the   variable "x_min" at line 140 of the "e-braille-tales.py" Python code</b>, in order to initially calibrate the code to your Perkins Brailler/scanner       combination. Remember to <b>always set the left margin of the Perkins Brailler to its minimum setting</b> (see explanation above in the                   "Dependencies / Limitations" section). Go ahead and open the JPEG file with segmentation results (green rectangles) in a photo editing software such as   GIMP. Take note of the pixel at which the braille character starts along the x axis (in landscape mode) and update the value at line 140 of the "e-       braille-tales.py" Python code. Please refer to Figure 1 for more details on this step.  




This Python code enables you to see the segmentation results (the green rectangles delimiting
the individual characters on the typewritten image) and then write a ".txt" file with the correct labels for each rectangle. The mapping
of every rectangle to a label will allow to generate a dataset of character images with their corresponding labels. The typewriter
page images overlaid with the character rectangles are stored in the "Page image files with rectangles" folder, which is created
automatically by the code.

You might need to <b>alter the values</b> of the variables "<b>character_width</b>" (default value of 55 pixels for 8 1/2" x 11" typewritten pages 
scanned at a resolution of 600 dpi) and "<b>spacer_between_characters</b>" (default value of 5 pixels), as your typewriter may have a different typeset than those of my typewriters (those two default parameters were suitable for both my <i>2021 Royal Epoch</i> and <i>1968 Olivetti Underwood Lettera 33</i> typewriters). Also, if your typewriter has a lot of ghosting (faint outline of the preceding character) or if the signal to noise ratio is elevated (because of high ink loading on the ribbon leading to lots of ink speckling on the page), the segmentation code might pick up the ghosting or noise as characters. As a result, you could then end up with staggered character rectangles. In the presence of dark typewritten text you should decrease the segmentation sensitivity (increase the number of non-white y pixels required for a given x coordinate in order for that x coordinate to be included in the segmentation). That is to say that on a fresh ribbon of ink, you should increase the value of 3 (illustrated below) to about 6 (results will vary based on your typewriter's signal to noise ratio) in the line 57 of "get_predictions.py" in order to avoid including unwanted noise in the character rectangles. 
```
x_pixels = np.where(line_image >= 3)[0] 
```
When your typewritten text gets fainter, change that digit back to 3 to make the segmentation more sensitive (to avoid omitting characters). These parameters ("character_width", "spacer_between_characters" and "line_image >= 3" should be adjusted in the same way in all the Python code files (except "train_model.py", where they are absent) to ensure consistent segmentation in all steps of the process.

![Image txt file processing](https://github.com/LPBeaulieu/TintypeText/blob/main/txt%20file%20example.jpg)<hr>
The image above illustrates the format of the ".txt" file listing all of the character rectangle labels. In the first line, you can note that four of the characters are labeled as "@", which maps to the category "to be deleted". The three letters (C, X and I) have significant ink splattering and will not be included in the training data, as they are not representative of these characters. The fourth "@" on the first line corresponds to an artifact (some noise was above the filtering threshold and was picked up as a character). We also do not want to include it in the training data. The "lesser than" symbol highlighted in yellow on line 11 in the ".txt" file corresponds to an "empty" rectangle, which is mapped to the "space" category in the "Dataset" folder. The very last line of the typewriter scan image contains two typos (two characters overlaid with a hashtag symbol). They are represented by a "~" symbol in the ".txt" file on line 19. All the other character rectangles are represented by their own characters in the ".txt" file. 
<br><br>
I

  <br><b>File 4: "get_predictions.py"</b>- This code will perform OCR on JPEG images of scanned typewritten text (at a resolution of 600 dpi)
  that you will place in the folder "OCR Raw Data". 
  
  <b>Please note that all of the JPEG file names in the "OCR Raw Data" folder must contain at least one hyphen ("-") in order for the code
  to properly create subfolders in the "OCR Predictions" folder. These subfolders will contain the rich text format (RTF) OCR conversion documents.</b> 
  
  The reason for this is that when you will scan a multi-page document in a multi-page scanner, you will provide you scanner with a file root name (e.g. "my_text-") and the scanner will number them automatically (e.g."my_text-.jpg", "my_text-0001.jpg", "my_text-0002.jpg", "my_text-"0003.jpg", etc.) and the code would then label the subfolder within the "OCR Predictions" folder as "my_text". The OCR prediction results for each page will be added in sequence to the "my_text.rtf" file within the "my_text" subfolder of the "OCR Predictions" folder. Should you ever want to repeat the OCR prediction for a set of JPEG images, it would then be important to remove the "my_text" subfolder before running the "get_predictions.py" code once more, in order to avoid appending more text to the exising "my_text.rtf" file.

If you changed the name of your deep learning model, or if you are using one of the models that I trained, you will to update the model name within the "get_predictions.py" code. That is to say that you will need to change "typewriter_OCR_cnn_model" for the name of your model in line 174 of "get_predictions.py":
               
```              
learn = load_learner(cwd + '/typewriter_OCR_cnn_model')
```
               
As mentioned above, since fresh typewriter ink ribbons lead to darker text and more ink speckling on the page, in the presence of dark typewritten text you should decrease the segmentation sensitivity (increase the number of non-white y pixels required for a given x coordinate in order for that x coordinate to be included in the segmentation). That is to say that on a fresh ribbon of ink, you should increase the value of 3 (illustrated below) to about 6 (results will vary based on your typewriter's signal to noise ratio) in the line 56 of "get_predictions.py" in order to avoid including unwanted noise in the character rectangles. 
```
x_pixels = np.where(line_image >= 3)[0] 
```
When your typewritten text gets fainter, change that digit back to 3 to make the segmentation more sensitive (to avoid omitting characters).

        
  <br><b>And that's it!</b> You're now ready to convert your typewritten manuscript into digital format! You can now type away at the cottage or in the park without worrying about your laptop's battery life 
  and still get your document polished up in digital form in the end! 🎉📖
  
  
## ✍️ Authors <a name = "author"></a>
- 👋 Hi, I’m Louis-Philippe!
- 👀 I’m interested in natural language processing (NLP) and anything to do with words, really! 📝
- 🌱 I’m currently reading about deep learning (and reviewing the underlying math involved in coding such applications 🧮😕)
- 📫 How to reach me: By e-mail! LPBeaulieu@gmail.com 💻


## 🎉 Acknowledgments <a name = "acknowledgments"></a>
- Hat tip to [@kylelobo](https://github.com/kylelobo) for the GitHub README template!




<!---
LPBeaulieu/LPBeaulieu is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
