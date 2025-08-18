import React, { useState } from "react";
import axios from "axios";

//import TestUpload from "./components/TestUpload"; (TEST UPLOAD COMPONENT)
import DarkModeToggle from "./components/DarkModeToggle";
import FontToggle from "./components/FontToggle";
import ImageUpload from "./components/ImageUpload";
import ResultCard from "./components/ResultCard";
import GamificationStatus from "./components/GamificationStatus";

import styles from "./styles/App.module.css";

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [dyslexiaFont, setDyslexiaFont] = useState(false);
  const [result, setResult] = useState(null);
  const [xp, setXp] = useState(0);
  const [level, setLevel] = useState(1);
  const [streak, setStreak] = useState(0);

  // -------------------------------
  // Handles image upload & prediction
  // -------------------------------
  const handlePredict = async (imageFile) => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append("file", imageFile);

    try {
      // Use environment variable for backend URL
      const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

      const res = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 
          "Content-Type": "multipart/form-data" 
        },
      });

      const data = res.data;

      // Update result card
      setResult({
        disease: data.disease,
        confidence: (data.confidence * 100).toFixed(2),
        xpGained: data.xp,
        streak: data.streak,
        badges: data.badges,
        fact: data.fact,
      });

      // Update gamification status
      setXp(data.xp);
      setLevel(Math.floor(data.xp / 100) + 1);
      setStreak(data.streak);
    } catch (err) {
      console.error("Error connecting to backend:", err);
      alert("Failed to connect to backend. Make sure Flask is running!");
    }
  };

  return (
    <div
      className={`${styles.app} ${darkMode ? styles.dark : ""} ${
        dyslexiaFont ? styles.dyslexiaFont : ""
      }`}
    >
      <header className={styles.header}>
        <h1>Plant Disease Detector</h1>
        <div className="flex gap-4">
          <DarkModeToggle darkMode={darkMode} setDarkMode={setDarkMode} />
          <FontToggle dyslexiaFont={dyslexiaFont} setDyslexiaFont={setDyslexiaFont} />
        </div>
      </header>

<main className="mt-6">
  <GamificationStatus xp={xp} level={level} streak={streak} />

  <ImageUpload onPredict={handlePredict} />

  {result && <ResultCard result={result} />}
</main>
    </div>
  );
}
