# Hand-Gesture-Recognition

## A machine learning basesd hand gesture classifier

![alt text](./Documentation/logo.jpg)

<div align="center">

[![GitHub contributors](https://img.shields.io/github/contributors/Waer1/Hand-Gesture-Recognition)](https://img.shields.io/github/Waer1/Hand-Gesture-Recognition/contributors)
[![GitHub issues](https://img.shields.io/github/issues/Waer1/Hand-Gesture-Recognition)](https://github.com/Waer1/Hand-Gesture-Recognition/issues)
[![GitHub forks](https://img.shields.io/github/forks/Waer1/Hand-Gesture-Recognition)](https://github.com/Waer1/Hand-Gesture-Recognition/network)
[![GitHub stars](https://img.shields.io/github/stars/Waer1/Hand-Gesture-Recognition)](https://github.com/Waer1/Hand-Gesture-Recognition/stargazers)
[![GitHub license](https://img.shields.io/github/license/Waer1/Hand-Gesture-Recognition)](https://github.com/Waer1/Hand-Gesture-Recognition/blob/master/LICENSE)

</div>
<details open="open">
<summary>
<h2 style="display:inline">📝 Table of Contents</h2>
</summary>

- [Built With](#built-with--)
- [Getting started](#getting-started)
- [Description](#description)
- [Project Pipeline](#pipeline)
</details>
<hr>

<h2 href="#BuiltWith">Built With </h2>
 <ul>
  <li><a href="https://www.python.org/">Python</a></li>
  <li><a href="https://pypi.org/project/opencv-python/">CV2</a></li>
  <li><a href="https://docs.python.org/3/library/pickle.html">Pickle</a></li>
   <li><a href="https://docs.python.org/3/library/argparse.html">argparse</a></li>
  <li><a href="https://numpy.org/">NumPy</a></li>
  <li><a href="https://scikit-image.org/">Skimage</a></li>
  <li><a href="https://scikit-learn.org/">Skilearn</a></li>
  <li><a href="https://docs.python.org/3/library/os.html">OS Python Library</a></li>
  <li><a href="https://docs.python.org/3/library/time.html">Time Python Library</a></li>
 </ul>
<hr>

<h2 href="#GettingStarted">Getting Started</h2>
<blockquote>
  <p>This is a list of needed steps to set up your project locally, to get a local copy up and running follow these instructions.
 </p>
</blockquote>
<ol>
  <li><strong><em>Clone the repository</em></strong>
    <div>
        <code>$ git clone https://github.com/Waer1/Hand-Gesture-Recognition</code>
    </div>
  </li>
  <li><strong><em>Install Pip and Python</em></strong>
    <div>
        <h4>Follow this article to install Pip and Python <a href="https://phoenixnap.com/kb/install-pip-windows">Install Pip and Python</a></h4>
    </div>
  </li>
  <li><strong><em>Install dependencies</em></strong>
    <div>
        <h4>Please refer to previous Built With list to install all needed libraries</h4>
    </div>
  </li>
  <li><strong><em>Train the model</em></strong>
    <div>
        <code>$ py .\FinalCode\main.py --feature 0  --model 0 </code>
        <br>
        <strong><em>Note: You should put the images under a folder named <code>Dataset</code> under the main folder <code>Hand-Gesture-Recognition</code></em></strong>
    </div>
  </li>
  
  <li><strong><em>Test the model and performance analysis</em></strong>
    <div>
        <code>$ py .\FinalCode\app.py --feature 0 </code>
        <br>
        <strong><em>Note: You should put the images under a folder named <code>data</code> under the main folder <code>Hand-Gesture-Recognition</code></em></strong>
    </div>
  </li>

</ol>
<hr>

<h2 href="#Description">Description</h2>

<blockquote style="font-size: 15px; font-weight: 500">
  <p >
    Hand gesture recognition has become an important area of research due to its potential applications in various fields such as robotics, healthcare, and gaming. The ability to recognize and interpret hand gestures can enable machines to understand human intentions and interact with them more effectively. For instance, in healthcare, hand gesture recognition can be used to control medical equipment without the need for physical contact, reducing the risk of infection. In gaming, it can enhance the user experience by allowing players to control games using hand gestures instead of traditional controllers. Overall, hand gesture recognition has immense potential to revolutionize the way we interact with machines and improve our daily lives.
  </p>

</blockquote>

<hr>

<h2 href="#Pipeline">Project Pipeline</h2>

<blockquote style="font-size: 15px; font-weight: 500">
  <p >
    You can also look at it at <code>./Documentation/Project Pipeline.drawio</code>
  </p>

</blockquote>

![alt text](./Documentation/ProjectPipeline.png)

<hr>

<h2 href="#Preprocessing">Preprocessing Module</h2>

<blockquote style="font-size: 15px; font-weight: 500">
  <p >
    Our main problem at this stage is the fingers shadows due to poor imaging conditions, so, we overcome over that by applying multiple steps for every input image:
    <ol>
      <li>
        <strong><em>Convert the input image to YCrCb color space.</em></strong>
      </li>
      <li>
        <strong><em>Get the segmented image using 'range_segmentation' method as follows:</em></strong>
        <ol>
          <li>
            Segment the image based on the lower and upper bounds of skin color defined in YCrCb color space.
          </li>
          <li>
            Apply morphological operations to remove noise.
          </li>
          <li>
            Find the contours in the binary segmented image, get the contour with the largest area.
          </li>
          <li>
            Create a blank image to draw and fill the contours.
          </li>
          <li>
            Draw the largest contour on the blank image and fill the contour with white color.
          </li>
          <li>
            Return the image with the largest contour drawn on it.
          </li>
        </ol>
      </li>
      <li>
        <strong><em>Apply thresholding on cr and cb components.</em></strong>
      </li>
      <li>
        <strong><em>Apply the following formula bitwise <code>(cr || cb) && range_segmented_image.</code></em></strong>
      </li>
      <li>
        <strong><em>Apply morphological operations to remove noise.</em></strong>
      </li>
      <li>
        <strong><em>Get the image with the largest contour area.</em></strong>
      </li>
      <li>
        <strong><em>Apply histogram equaliztion to make the image more clear.</em></strong>
      </li>
      <li>
        <strong><em>Cut the greyscale image around the largest contour.</em></strong>
      </li>
      <li>
        <strong><em>Resize the image to small size to reduce extracted features array length.</em></strong>
      </li>
    </ol>
  </p>

</blockquote>

<hr>

<h2 href="#Features-Extraction">Features Extraction Module</h2>

<blockquote style="font-size: 15px; font-weight: 500">
  <p>
    We tried multiple algorithms: SURF, SIFT, LBP, and HOG. And we selected HOG (Histogram of Oriented Gradients) for providing high accuracy.
    <br>
    <br>
    HOG algorithm is a computer vision technique used to extract features from images. It works by dividing an image into small cells and computing the gradient orientation and magnitude for each pixel within the cell. The gradient orientations are then binned into a histogram, which represents the distribution of edge orientations within that cell.
    <br>
    <br>
    The HOG algorithm has high accuracy because it is able to capture important information about the edges and contours in an image. This information can be used to identify objects or patterns within the image, even if they have varying lighting conditions.
    <br>
    <br>
    We apply HOG algorithm using hog() from skimage.feature, and with making sure that all features vectors are with the same length by padding zeros.
  </p>

</blockquote>
