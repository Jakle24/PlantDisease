import React, { useState } from 'react';
import DarkModeToggle from './components/DarkModeToggle';
import FontToggle from './components/FontToggle';
import ImageUpload from './components/ImageUpload';
import ResultCard from './components/ResultCard';
import GamificationStatus from './components/GamificationStatus';
import styles from './styles/App.module.css';
import axios from "axios";

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [dyslexiaFont, setDyslexiaFont] = useState(false);
  const [result, setResult] = useState(null);
  const [xp, setXp] = useState(0);
  const [level, setLevel] = useState(1);
  const [streak, setStreak] = useState(0);

const handlePredict = async (imageFile) => {
  if (!imageFile) return;

  const formData = new FormData();
  formData.append("file", imageFile);

  try {
    const res = await axios.post("http://localhost:5000/predict", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    const data = res.data;

    // Update frontend state with backend response
    setResult({
      disease: data.disease,
      confidence: (data.confidence * 100).toFixed(2), // convert to %
      xpGained: data.xp, // cumulative XP
      streak: data.streak,
      badges: data.badges,
      fact: data.fact,
    });

    // Update XP / level / streak
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
      className={`${styles.app} ${darkMode ? styles.dark : ''} ${
        dyslexiaFont ? styles.dyslexiaFont : ''
      }`}
    >
      <header className={styles.header}>
        <h1>Plant Disease Detector</h1>
        <div>
          <DarkModeToggle darkMode={darkMode} setDarkMode={setDarkMode} />
          <FontToggle dyslexiaFont={dyslexiaFont} setDyslexiaFont={setDyslexiaFont} />
        </div>
      </header>
      <main>
        <GamificationStatus xp={xp} level={level} streak={streak} />
        <ImageUpload onPredict={handlePredict} />
        {result && <ResultCard result={result} />}
      </main>
    </div>
  );
}
