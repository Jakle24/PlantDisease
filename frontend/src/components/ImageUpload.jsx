import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

export default function ImageUpload({ onPredict }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const backendUrl = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
  const submittingRef = useRef(false); // guard against duplicate submissions

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handleFileChange = (e) => {
    setErrorMsg(null);
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg(null);

    if (submittingRef.current) return; // prevent double submit
    if (!file) {
      setErrorMsg("Please select an image first.");
      return;
    }

    submittingRef.current = true;
    setLoading(true);

    console.log("Uploading file:", { name: file.name, size: file.size, type: file.type });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(`${backendUrl}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = res.data;

      // Pass only prediction result to App
      onPredict({
        disease: data.disease,
        confidence: `${data.confidence.toFixed(2)}%`,
        xpGained: data.xp,
        streak: data.streak,
        badges: data.badges,
        fact: data.fact,
      });
    } catch (err) {
      console.error("Error connecting to backend:", err);
      setErrorMsg(
        err?.response?.data?.error ||
        err?.message ||
        "Failed to connect to backend. Make sure Flask is running at http://localhost:5000"
      );
    } finally {
      setLoading(false);
      submittingRef.current = false;
    }
  };

  return (
    <div className="mt-4">
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit" disabled={loading} className="ml-2 px-4 py-2 bg-green-600 text-white rounded">
          {loading ? "Scanning..." : "Scan Plant"}
        </button>
      </form>

      {previewUrl && (
        <div style={{ marginTop: 12 }}>
          <strong>Preview:</strong>
          <div style={{ marginTop: 8 }}>
            <img src={previewUrl} alt="preview" style={{ maxWidth: "320px", maxHeight: "320px", borderRadius: 8 }} />
          </div>
        </div>
      )}

      {errorMsg && (
        <div style={{ marginTop: 12, color: "#a00" }}>
          <strong>Error:</strong> {errorMsg}
        </div>
      )}

      {file && <p className="mt-2">Selected file: {file.name}</p>}
    </div>
  );
}
