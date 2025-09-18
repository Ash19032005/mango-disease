import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://127.0.0.1:8000'; // Match your backend URL

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [modelName, setModelName] = useState('model1');
  const [previewUrl, setPreviewUrl] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionSuccess, setPredictionSuccess] = useState(false); // New state for success animation

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPredictions(null); // Clear previous predictions
      setError(null);
      setPredictionSuccess(false); // Reset success state
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setPredictions(null); // Clear previous predictions when new prediction starts
    setError(null); // Clear previous errors
    setPredictionSuccess(false); // Reset success state

    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const response = await axios.post(`${API_URL}/predict?model_name=${modelName}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPredictions(response.data.predictions);
      setPredictionSuccess(true); // Set success state
      // Optional: hide success animation after a few seconds
      // setTimeout(() => setPredictionSuccess(false), 3000); 
    } catch (err) {
      console.error("Prediction failed:", err);
      setError("Failed to get prediction. Please try again.");
      setPredictions(null);
    } finally {
      setLoading(false);
    }
  };

  const formatPredictions = () => {
    if (!predictions) return null;
    return Object.entries(predictions).map(([label, confidence]) => (
      <li key={label}>
        <strong>{label}:</strong> { (confidence * 100).toFixed(2) }%
      </li>
    ));
  };

  return (
    <div className="App">
      <h1>Mango Disease Detector</h1>
      <p>Upload an image of a mango and select a model to predict the disease.</p>

      <div className="controls">
        <div className="file-upload">
          <label>
            Upload Image:
            <input type="file" onChange={handleFileChange} accept="image/*" />
          </label>
        </div>

        <div className="model-select">
          <label>
            Select Model:
            <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              <option value="model1">Resnet-50</option>
              <option value="model2">VGG-16</option>
              <option value="model3">MobileVnet</option>
              <option value="model4">InceptionV3</option>
            </select>
          </label>
        </div>
        
        <button onClick={handlePredict} disabled={loading || !selectedFile}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      <div className="results-container">
        {/* Conditional rendering for image preview or placeholder */}
        <div className="image-display-area">
          {previewUrl ? (
            <div className="image-preview">
              <h2>Uploaded Image</h2>
              <img src={previewUrl} alt="Preview" />
            </div>
          ) : (
            <div className="upload-placeholder">
              <img src="/images/upload-placeholder.png" alt="Upload an image" />
              <p>Upload a mango leaf image to get started!</p>
            </div>
          )}
        </div>

        {error && <div className="error">{error}</div>}

        {/* Conditional rendering for loading spinner, success animation, or predictions */}
        {loading && (
          <div className="loading-indicator">
            <img src="/images/loading-spinner.png" alt="Loading..." />
            <p>Analyzing image...</p>
          </div>
        )}

        {predictionSuccess && !loading && predictions && (
          <div className="prediction-success-animation">
            <img src="images/prediction-success.png" alt="Prediction Complete!" />
            <p>Prediction Complete!</p>
          </div>
        )}

        {predictions && !loading && ( // Display predictions only if not loading and available
          <div className="predictions-list">
            <h2>Prediction Results</h2>
            <ul>
              {formatPredictions()}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;