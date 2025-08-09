import React from 'react';
import styles from '../styles/FontToggle.module.css';

export default function FontToggle({ dyslexiaFont, setDyslexiaFont }) {
  return (
    <button
      className={styles.toggleButton}
      onClick={() => setDyslexiaFont(!dyslexiaFont)}
      aria-label="Toggle dyslexia-friendly font"
    >
      {dyslexiaFont ? '🔤 Regular Font' : '🔠 Dyslexia-Friendly Font'}
    </button>
  );
}
