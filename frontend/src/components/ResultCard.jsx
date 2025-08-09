import React from 'react';
import styles from '../styles/ResultCard.module.css';

export default function ResultCard({ result }) {
  return (
    <div className={styles.card} aria-live="polite">
      <h2>
        {result.plant} — {result.disease}
      </h2>
      <p>Confidence: {result.confidence}%</p>
      <p className={styles.funFact}>💡 {result.funFact}</p>
      <div className={styles.xpContainer}>
        <p>XP Gained: {result.xpGained}</p>
        <div className={styles.progressBar}>
          <div
            className={styles.progress}
            style={{ width: `${result.progressPercent}%` }}
          ></div>
        </div>
        <p>Level: {result.level}</p>
      </div>
    </div>
  );
}
