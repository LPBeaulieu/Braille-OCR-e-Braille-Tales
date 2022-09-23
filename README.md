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

<p align="left"> <b>e-Braille Tales</b> is a tool enabling you to convert scanned braille pages (in JPEG image format and typed on a Perkins Brailler) into Portable Embosser Format (PEF) digitized braille and rich text format (RTF) documents, complete with formatting elements such as alignment, paragraphs, <u>underline</u>, <i>italics</i>, <b>bold</b> and <del>strikethrough</del>, basically allowing you to include any formatting encoded by RTF commands or braille typeform indicators.</p>
<p align="left"> A neat functionality of <b>e-Braille Tales</b> is that the typos (sequence of at least two successive full braille cells)
  automatically get filtered out, and do not appear in the final RTF text nor in the PEF file. The PEF file can in turn be used to print out copies of your work on a braille embosser, or to read them electronically using a refreshable braille display.
  
  - My <b>deep learning model</b> for the Perkins Brailler along with the dataset and other useful information may be found on my Google Drive at the following link: https://drive.google.com/drive/folders/1RNGUoBJOSamYOaO7ElFBeWIRVpHtlQpd?usp=sharing. 
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
  
- When typing text on the Perkins Brailler, unless a space is included at the end of a line or at the beginning of the next line, the last word on the line will be merged with the first characters on the next one, up to the next space. As such, the <b>"line continuation without space" braille symbol ("⠐") is not required and should be avoided</b>, as it could be confused with other braille characters, such as initial-letter contractions. However, line continuations with a space ("⠐⠐") can be used without problem in this application.

- In this application a space needs to be included after any RTF command (even though the RTF specifications state that it is an optional space). The reason for this is that when the code is transcribing the braille into printed English, it often needs to determine if any given braille character stands alone. A braille character that stands alone means that it is flanked by characters such as empty braille cells ("⠀") or dashes, but not by a       braille character mapping to a letter or number, such that can be found at the end of every RTF command. In other words, <b>you must include a space after any RTF commands</b>. Here is an example: "This requirement \strike strikes \strike0 me as being important!", which in braille would be written as follows: "⠠⠹⠀⠗⠑⠟⠥⠊⠗⠑⠰⠞⠀⠸⠡⠎⠞⠗⠊⠅⠑⠀⠎⠞⠗⠊⠅⠑⠎⠀⠸⠡⠎⠞⠗⠊⠅⠑⠼⠚⠀⠍⠑⠀⠵⠀⠆⠬⠀⠊⠍⠏⠕⠗⠞⠁⠝⠞⠖".

- Importantly, <b>the pages must be scanned with the left margin placed on the flatbed scanner in such a way that the shadows produced by the 
  scanner light will face away from the left margin</b> (the shadows will face the right margin of the page, when the page is viewed in landscape mode). 
  This is because the non-white pixels actually result from the presence of shadows, the orientation of which plays a major role in image segmentation (determining the x and y coordinates of the individual characters) and optical character recognition (OCR). For best results, the braille document 
  should be <b>typed on white braille paper or cardstock and scanned as grayscale images on a flatbed scanner at a 300 dpi resolution with the paper size setting of the scanner set to letter 8 1/2" x 11" (A4)</b>. The darkness settings of the scanner might also need to be adjusted to acheive an optimal braille shadow to noise ratio. When scanning the braille pages, <b>some weight (such as 6-inch metal rulers) should be placed on the back of the braille pages to prevent them from sliding on the glass of the flatbed scanner</b>. The pages tend to move around when closing the lid, as there is very little friction keeping them in place, since their only points of contact with the glass are the embossed braille dots. Should the page move out of line, then the segmentation results could be adversely affected. <b>To ensure that the segmentation has proceeded adequately, the segmentation result image (scanned image overlaid with green character rectangles) for every scanned page of the braille document should be quickly inspected</b>. These         images are generated by the code and stored in the "Page image files with rectangles" folder, which is created automatically by the code.  
  
- <b>The left margin on the Perkins Brailler should be set at its minimal setting</b> in order to maximize the printable space on the page and to always provide the same reference point to the code for the segmentation step. <b>The pixel "x_min", at which the code starts cropping characters on every line, needs to be entered manually in the code, as you initially calibrate the code to your own brailler and scanner combination</b>. In my case, the value of the   variable "x_min" is set to 282 pixels in line 140 of the Python code "e-braille-tales.py". After running the code on a scanned braille text image of yours, you could then open the JPEG image overlaid with green character rectangles (see Figure 1 below) in a photo editing software such as GIMP, in order to locate the pixel value along the x axis (in landscape mode) at which the segmentation should start in each line. 

 - Every brailled line should have braille characters that when taken together contain at least three dots per braille cell row in order to be properly detected. Should a line only contain characters that do not have dots in one or more of the three braille cell rows, you could make up for the missing dots by using at least two successive full braille cells ("⠿") before or after the text (for example: "⠿⠿⠿YOUR SHORT BRAILLE LINE HERE"), which will be interpreted by the code as a typo, and    will not impact the meaningful text on the line in the final Rich Text Format (RTF) and Portable Embosser Format (PEF) files.
 
 
## 🏁 Getting Started <a name = "getting_started"></a>

The following instructions will be provided in great detail, as they are intended for a broad audience and will allow to run a copy of <b>e-Braille Tales</b> on a local computer. As the steps 1 through 8 described below are the same as for my other project <b>Tintype Text</b>  (https://github.com/LPBeaulieu/Typewriter-OCR-TintypeText), a link is provided here to an instructional video explaining how to set up <b>Tintype Text</b>: https://www.youtube.com/watch?v=FG9WUW6q3dI&list=PL8fAaOg_mhoEZkbQuRgs8MN-QSygAjdil&index=2.

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

<b>Step 3</b>- Activate the <i>env</i> virtual environment <b>(you will need to do this step every time you use the Python code file)</b> 
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

<b>Step 7</b>- Install <b>alive-Progress</b> (Python module for a progress bar displayed in command line):
```
pip install alive-progress
```

<b>Step 8</b>- Create the folder "OCR Raw Data" in your working folder:
```
mkdir "OCR Raw Data" 
```
<b>Step 9</b>- You're now ready to use <b>e-Braille Tales</b>! 🎉

## 🎈 Usage <a name="usage"></a>
The "e-braille-tales.py" Python code converts JPEG braille text scans into printed English in the form of a Rich Text Format (RTF) document and digitized braille as a Portable Embosser Format (PEF) file. In addition to the RTF and PEF files, the code will generate a braille text file (".txt") containing the OCR results before transcription to printed English, so that you could revisit the text in braille form. Each page of this ".txt" file will line up with the pages written on the Perkins Brailler and will be separated from one another by two carriage returns, to ensure easy navigation throughout the document. You can find instructions on how to use <b>e-Braille Tales</b> on my YouTube channel: https://www.youtube.com/watch?v=U8-s8eQXInI.<br>

- In order to submit a scanned braille text page to the code, you will need to <b>place the JPEG image in the "OCR Raw Data" subfolder of your working     folder</b>, which you created at step 8 of the "Getting Started" section.

- <b>Please note that all of the JPEG file names in the "OCR Raw Data" folder must contain at least one hyphen ("-") in order for the code
  to properly create subfolders in the "OCR Predictions" folder.</b> These subfolders will contain the RTF document, along with the PEF and ".txt" braille files. The reason for this is that when you will scan a multi-page document, you will provide your scanner with a file root name (e.g. "my_text-") and the scanner will number them automatically (e.g."my_text-.jpg", "my_text-0001.jpg", "my_text-0002.jpg", "my_text-"0003.jpg", etc.) and   the code would then label the subfolder within the "OCR Predictions" folder as "my_text". The OCR prediction results for each page will be added in sequence to the "my_text.txt" file within the "my_text" subfolder of the "OCR Predictions" folder. Should you ever want to repeat the OCR prediction for a set of JPEG images, it would then be important to remove the "my_text" subfolder before running the "get_predictions.py" code once more, in order to avoid appending more text to the existing "my_text.txt" file.
  
- Then, run the "e-braille-tales.py" Python script by opening the command line from your working folder, such that you will already be in the correct path and copy and paste the following in command line:  
```
python3 e-braille-tales.py
```

- The first thing that the code will do is perform segmentation (determine the x and y coordinates of every braille character). The segmentation results are visible in the "Page image files with rectangles" folder, which is created automatically by the code. You might need to <b>adjust the value of the variable "x_min" at line 140 of the "e-braille-tales.py" Python code</b>, in order to initially calibrate the code to your Perkins Brailler/scanner combination. Remember to <b>always set the left margin of the Perkins Brailler to its minimum setting</b> (see explanation above in the                   "Dependencies / Limitations" section). Go ahead and open the JPEG file with segmentation results (green rectangles) in a photo editing software such as GIMP. Take note of the pixel at which the braille character starts along the x axis (in landscape mode) and update the value at line 140 of the "e-braille-tales.py" Python code. You should only need to find the pixel value of "x_min" and update it in the code once, as illustrated in Figure 1. 

![Image txt file processing](https://github.com/LPBeaulieu/Braille-OCR-e-Braille-Tales/blob/main/Figure%201%20(explanation%20of%20x_min).png)<hr>
<b>Figure 1</b>: The pixel along the x-axis (in landscape mode) at which segmentation should start on every line can be found by opening the scanned braille JPEG image in a photo editing software such as GIMP and locating the pixel closest to the left margin (see red arrows), here "x_min" is set to 282 pixels.

- Alternatively, it is possible to resubmit the text (".txt") file to the "e-braille-tales.py" Python code once you have made modifications to it. The braille text will be extracted from the ".txt" file and the carriage returns that were introduced to facilitate proofreading will be automatically removed by the code, if still present. Simply place the corrected ".txt" file in the "OCR Raw Data" subfolder of your working folder and include the name of your text file when running the Python code, as follows:
```
python3 e-braille-tales.py "my_text_file_name.txt"
```
- When providing Python with the name of your file (and placing the text file in the "OCR Raw Data" folder), the OCR step will be circumvented and your braille text will be converted to the RTF and PEF files. You can continue this process until all mistakes have been dealt with.
 
- The following RTF commands are automatically converted into PEF tags by the code and are transcribed from braille to English RTF commands in the RTF file: 

  - The braille equivalent of the tab RTF command "\tab" ("⠸⠡⠞⠁⠃") will be changed to two successive empty braille cells ("⠀⠀").
  - A line break RTF command "\line" ("⠸⠡⠇⠔⠑") will be converted into a line break (\</row>\<row> PEF tags).
  - New paragraph RTF commands "\par" ("⠸⠡⠏⠜") will be mapped to a line break (\</row>\<row> PEF tags) followed by two successive empty braille cells, as new paragraphs in braille documents are typically started by two empty braille cells that serve as a tab. Similarly, in the RTF document, any braille new paragraph RTF commands "\par" ("⠸⠡⠏⠜") will be switched to "\par \tab" to add a tab at the start of every new paragraph.
  - The page break RTF commands "\page" ("⠸⠡⠏⠁⠛⠑") are changed for a line break, followed by a page break (\</row>\</page>\<page>\<row> PEF tags).
  - New section RTF commands "\sbkpage" ("⠸⠡⠎⠃⠅⠏⠁⠛⠑") are swapped out for a line break, followed by a page and section break 
    (\</row>\</page>\</section>\<section>\<page>\<row> PEF tags).

  For an in-depth explanation of all the most common RTF commands and escapes, please consult: https://www.oreilly.com/library/view/rtf-pocket-guide/9781449302047/ch01.html.

  These are the only RTF commands that are automatically removed from the braille text and converted into PEF tags. All other RTF commands (if present) will be carried over in braille form into the PEF file and could be removed manually afterwards. However, as braille already encompasses typeform  indicators for symbols, words and passages written in caps, italics, bold, underline or script (font size of 28), as well as symbols in superscript or   subscript, there should be limited need to resort to other RTF commands than those listed above. 
  
- When using grade I ("⠰") or numeric ("⠼") indicators, these should be placed directly in front of the characters they will be affecting. The next order of priority is the capitalization indicators ("⠠"), followed by the other typeform indicators (bold, italics, underline, script) and finally by superscript "⠰⠔" or subscript "⠰⠢" indicators. 
 
     
<br><b>And that's it!</b> You're now ready to convert your braille manuscript into digital format! If you are close to someone who is visually impaired and would like to help them find meaningful work through technology, or maybe if you are only sprucing up your braille skills in preparation for the Zombie Apocalypse (lol) then this app is for you! 🎉📖
  
  
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
