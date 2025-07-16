import React, { useState } from "react";
import axios from "axios";

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const onFileChange = e => setFile(e.target.files[0]);

  const onSubmit = async () => {
    const data = new FormData();
    data.append("file", file);

    const res = await axios.post("http://localhost:5000/predict", data);
    setResult(res.data);
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl mb-4">Plant Disease Detector</h1>
      <input type="file" onChange={onFileChange} />
      <button onClick={onSubmit} className="ml-2 p-2 bg-blue-500 text-white">Analyze</button>

      {result && (
        <div className="mt-4">
          <p>Disease: <strong>{result.disease}</strong></p>
          <p>Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong></p>
        </div>
      )}
    </div>
  );
}
