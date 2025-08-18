import React, { useState } from "react";
import axios from "axios";

export default function TestUpload() {
  const [file, setFile] = useState(null);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const backendUrl = "http://localhost:5000"; // Replace if needed

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setResponse(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    try {
      const res = await axios.post(`${backendUrl}/predict`, formData);
      setResponse(res.data);
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Upload failed. Is Flask running at http://localhost:5000?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>Backend Test Upload</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit" disabled={loading} style={{ marginLeft: 10 }}>
          {loading ? "Uploading..." : "Upload"}
        </button>
      </form>

      {file && <p>Selected file: {file.name}</p>}

      {response && (
        <div style={{ marginTop: 20 }}>
          <h3>Backend Response:</h3>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
