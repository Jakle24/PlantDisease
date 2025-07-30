async function uploadImage() {
  const input = document.getElementById('fileInput');
  const file = input.files[0];

  if (!file) {
    alert("Please upload an image.");
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  const resultDiv = document.getElementById('result');

  if (data.error) {
    resultDiv.innerHTML = `❌ Error: ${data.error}`;
  } else {
    resultDiv.innerHTML = `🌱 Predicted Disease: <strong>${data.disease}</strong><br>🧪 Confidence: ${(data.confidence * 100).toFixed(2)}%`;
  }
}

function toggleTheme() {
  document.body.classList.toggle('dark');
}
