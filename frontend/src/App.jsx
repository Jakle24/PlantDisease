import React, { useState } from 'react';
import DarkModeToggle from './components/DarkModeToggle';
import FontToggle from './components/FontToggle';
import ImageUpload from './components/ImageUpload';
import ResultCard from './components/ResultCard';
import GamificationStatus from './components/GamificationStatus';
import styles from './styles/App.module.css';

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [dyslexiaFont, setDyslexiaFont] = useState(false);
  const [result, setResult] = useState(null);
  const [xp, setXp] = useState(0);
  const [level, setLevel] = useState(1);
  const [streak, setStreak] = useState(0);

  const handlePredict = async (imageFile) => {
    // Simulate API call delay
    await new Promise((r) => setTimeout(r, 1000));

    // Dummy response — replace with real API call later
    const fakeResponse = {
      plant: 'Rose',
      disease: 'Black Spot',
      confidence: 87,
      funFact: 'Black spot is one of the most common fungal diseases affecting roses.',
      xpGained: 15,
    };

    setResult(fakeResponse);

    // Update XP and level
    setXp(prev => prev + fakeResponse.xpGained);
    // Level up every 100 XP
    setLevel(prev => Math.floor((xp + fakeResponse.xpGained) / 100) + 1);
    // Increment streak (dummy logic)
    setStreak(prev => prev + 1);
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
