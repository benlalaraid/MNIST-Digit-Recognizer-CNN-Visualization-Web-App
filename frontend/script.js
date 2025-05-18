document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const clearBtn = document.getElementById('clearBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsContainer = document.getElementById('resultsContainer');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const predictedDigit = document.getElementById('predictedDigit');
    const confidenceValue = document.getElementById('confidenceValue');
    const probabilityBars = document.getElementById('probabilityBars');
    const inputImage = document.getElementById('inputImage');
    const conv1Output = document.getElementById('conv1Output');
    const pool1Output = document.getElementById('pool1Output');
    const conv2Output = document.getElementById('conv2Output');
    const pool2Output = document.getElementById('pool2Output');

    // API URL - adjust based on your FastAPI server
    const API_URL = 'http://localhost:8000';

    // Event Listeners
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    clearBtn.addEventListener('click', clearImage);
    analyzeBtn.addEventListener('click', analyzeImage);

    // Drag and drop events
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('drag-over');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('drag-over');
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files);
        }
    });

    // Functions
    function handleFileSelect(e) {
        if (e.target.files.length) {
            handleFiles(e.target.files);
        }
    }

    function handleFiles(files) {
        const file = files[0];
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            resultsContainer.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    function clearImage() {
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.style.display = 'none';
        resultsContainer.style.display = 'none';
    }

    async function analyzeImage() {
        if (!previewImage.src) {
            alert('Please select an image first');
            return;
        }

        // Show loading overlay
        loadingOverlay.style.display = 'flex';

        try {
            // Get the file from the input
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);

            // Send to API
            const response = await fetch(`${API_URL}/predict/`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Error analyzing image:', error);
            alert('Error analyzing image. Please try again.');
        } finally {
            // Hide loading overlay
            loadingOverlay.style.display = 'none';
        }
    }

    function displayResults(data) {
        // Show results container
        resultsContainer.style.display = 'block';

        // Display prediction
        predictedDigit.textContent = data.predicted_class;
        confidenceValue.textContent = `${(data.confidence * 100).toFixed(2)}%`;

        // Display probability bars
        probabilityBars.innerHTML = '';
        data.probabilities.forEach((prob, index) => {
            const bar = document.createElement('div');
            bar.className = 'probability-bar';
            bar.style.height = `${prob * 100}%`;
            bar.dataset.digit = index;
            bar.dataset.value = `${(prob * 100).toFixed(1)}%`;
            
            // Highlight the predicted class
            if (index === data.predicted_class) {
                bar.style.backgroundColor = '#27ae60';
            }
            
            probabilityBars.appendChild(bar);
        });

        // Display input image
        inputImage.innerHTML = '';
        const inputImgContainer = document.createElement('div');
        inputImgContainer.className = 'feature-map';
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${data.original_image}`;
        inputImgContainer.appendChild(img);
        inputImage.appendChild(inputImgContainer);

        // Display feature maps
        displayFeatureMaps(conv1Output, data.conv1_output);
        displayFeatureMaps(pool1Output, data.pool1_output);
        displayFeatureMaps(conv2Output, data.conv2_output);
        displayFeatureMaps(pool2Output, data.pool2_output);

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    function displayFeatureMaps(container, featureMaps) {
        container.innerHTML = '';
        featureMaps.forEach(base64Img => {
            const mapContainer = document.createElement('div');
            mapContainer.className = 'feature-map';
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${base64Img}`;
            mapContainer.appendChild(img);
            container.appendChild(mapContainer);
        });
    }
});
