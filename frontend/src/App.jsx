import React, { useState } from 'react';
import DarkModeToggle from './components/DarkModeToggle';
import FontToggle from './components/FontToggle';
import ImageUpload from './components/ImageUpload';
import ResultCard from './components/ResultCard';
import styles from './styles/App.module.css';

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [dyslexiaFont, setDyslexiaFont] = useState(false);
  const [result, setResult] = useState(null);

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
      level: 2,
      progressPercent: 65,
    };

    setResult(fakeResponse);
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
        <ImageUpload onPredict={handlePredict} />
        {result && <ResultCard result={result} />}
      </main>
    </div>
  );
}
