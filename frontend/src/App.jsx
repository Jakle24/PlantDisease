import React, { useState } from "react";
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
  const handlePredict = (data) => {
    // Update result card
    setResult({
      disease: data.disease,
      confidence: parseFloat(data.confidence).toFixed(2) + "%",
      xpGained: data.xpGained,
      streak: data.streak,
      badges: data.badges,
      fact: data.fact,
    });

    // Update gamification status
    const newXp = xp + data.xpGained;
    setXp(newXp);
    setLevel(Math.floor(newXp / 100) + 1);
    setStreak(data.streak);
  };

  return (
    <div className={`${styles.app} ${darkMode ? styles.dark : ""} ${dyslexiaFont ? styles.dyslexiaFont : ""}`}>
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
