import React from 'react';
import styles from '../styles/DarkModeToggle.module.css';

export default function DarkModeToggle({ darkMode, setDarkMode }) {
  return (
    <button
      className={styles.toggleButton}
      onClick={() => setDarkMode(!darkMode)}
      aria-label="Toggle dark mode"
    >
      {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
    </button>
  );
}
