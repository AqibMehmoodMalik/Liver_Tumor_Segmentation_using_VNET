<h1 align="center">🩺 Liver Tumor Segmentation</h1>
<p>Developed a V-Net-based deep learning model to segment liver tumors  from CT scans using the LiTS dataset. Pre-processed 3D medical images into 2D slices, applied data augmentation, and  evaluated performance using the Dice Coefficient.</p>
<h2 >📌 Project Overview</h2>
<p>
Liver cancer is one of the leading causes of cancer-related deaths worldwide. Early detection and precise localization of liver tumors are critical for diagnosis and treatment planning. 
This project focuses on building an automated system for <b>liver tumor segmentation</b> from CT scan images using deep learning models.
</p>

<p>
The system takes CT scan images as input and produces segmented masks that highlight tumor regions. This helps radiologists and doctors in faster and more accurate analysis.
</p>

<hr>
<h2>🎯 Objectives</h2>
<ul>
  <li>Automate liver tumor detection from CT scans.</li>
  <li>Accurately segment tumor regions.</li>
  <li>Reduce manual effort of radiologists.</li>
  <li>Provide a reliable AI-based diagnostic support system.</li>
</ul>

<hr>

<h2>🧠 Model & Approach</h2>
<ul>
  <li>Used Convolutional Neural Networks (CNN).</li>
  <li>Applied image preprocessing and normalization.</li>
  <li>Performed pixel-wise segmentation.</li>
  <li>Loss functions such as Dice Loss / Binary Cross Entropy used.</li>
  <li>Evaluation metrics include Accuracy, Dice Score, and IoU.</li>
</ul>

<h2>📁 Project Structure</h2>
<pre>
Final Code/
│
├── .venv/
├── .vscode/
├── requirements.txt
│
└── VNet/
    │
    ├── dataset/
    │
    ├── app.py
    ├── evaluate.py
    ├── final_pipline.py
    ├── predict.py
    ├── preprocess.py
    ├── train.py
    ├── utils.py
    ├── vnet.py
    ├── datasettraining.log
    ├── README.md
</pre>

<hr>
<h2>📂 Dataset</h2>
<p>
The dataset consists of liver CT scan images along with their corresponding ground truth masks.  
Each image has:
</p>
<ul>
  <li>Original CT scan image</li>
  <li>Liver mask</li>
  <li>Tumor mask</li>
</ul>

<p>
Dataset is preprocessed to resize images and normalize pixel values before training.
</p>

<hr>

<h2>⚙️ Features</h2>
<ul>
  <li>Automatic tumor segmentation</li>
  <li>Visual comparison of original image and predicted mask</li>
  <li>High accuracy on test data</li>
  <li>Easy to extend for other medical imaging tasks</li>
</ul>

<hr>

<h2>🛠️ Technologies Used</h2>
<ul>
  <li>Python</li>
  <li>TensorFlow / PyTorch</li>
  <li>OpenCV</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
  <li>Scikit-learn</li>
  </ul>


<h2>🚀 Installation</h2>
<pre>
git clone https://github.com/your-username/Liver-Tumor-Segmentation.git
cd Liver-Tumor-Segmentation
pip install -r requirements.txt
</pre>

<hr>

<h2>▶️ How to Run</h2>

<h3>1. Train the model</h3>
<pre>
python src/train.py
</pre>

<h3>2. Run the app </h3>
<pre>
python src/app.py
</pre>

<hr>

<h2>📊 Results</h2>
<p>
The model successfully segments liver tumors from CT scan images with good accuracy.  
Visual results show that predicted tumor masks closely match the ground truth masks.
</p>
<hr>
<h2>📸 Sample Output</h2>
<p>
Original Image → Predicted Mask → Tumor Highlighted Image
</p>

<hr>

<h2>🔮 Future Improvements</h2>
<ul>
  <li>Use advanced architectures like U-Net or Attention U-Net.</li>
  <li>Improve accuracy using data augmentation.</li>
  <li>Deploy as a web application.</li>
  <li>Support 3D CT scan volumes.</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>
<p>
<b>Aqib Mehmood</b><br>
AI / ML Engineer<br>
Pakistan
</p>

<hr>

<h2>📜 License</h2>
<p>
This project is for educational and research purposes only.
</p>
