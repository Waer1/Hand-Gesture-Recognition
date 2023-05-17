# Hand-Gesture-Recognition

<details open="open">
<summary>
<h2 style="display:inline">📝 Table of Contents</h2>
</summary>

- [Built With](#built-with-)
- [Getting started](#getting-started)
- [Performance Analysis and Results](#performance-analysis-and-results)
</details>
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

<h2 href="#Performance-Analysis-and-Results">Performance Analysis and Results</h2>

<blockquote style="font-size: 15px; font-weight: 500">
  <p>
    Our result is that the trained model predicted 83% of images correctly.
    <br>
    <br>
    You can run with your data under <code> data </code> folder under the main folder <code>Hand-Gesture-Recognition</code> the script <code>./FinalCode/app.py</code> to see output label and time taken for processing every image by using this command:
    <br>
    <code>$ py .\FinalCode\app.py --feature 0</code>
    <br>
    <strong><em>Note: You should see the output in 2 files: <code>results.txt</code> for output classes and <code>time.txt</code> for time taken to process the image after reading it and just after the prediction, every line in these 2 files is for one image.</em></strong>
  </p>

</blockquote>
