
import React, { useState } from "react";
import axios from "axios";

export default function ImageUpload({ onPredict }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const data = res.data;

      // Pass backend response to parent (App.jsx)
      onPredict({
        disease: data.disease,
        confidence: (data.confidence * 100).toFixed(2), // percentage
        xpGained: data.xp,
        streak: data.streak,
        badges: data.badges,
        fact: data.fact,
      });
    } catch (err) {
      console.error("Error connecting to backend:", err);
      alert("Failed to connect to backend. Make sure Flask is running!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4">
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button
          type="submit"
          disabled={loading}
          className="ml-2 px-4 py-2 bg-green-600 text-white rounded"
        >
          {loading ? "Scanning..." : "Scan Plant"}
        </button>
      </form>
      {file && <p className="mt-2">Selected file: {file.name}</p>}
    </div>
  );
}
