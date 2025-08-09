import React from 'react';
import styles from '../styles/GamificationStatus.module.css';

export default function GamificationStatus({ xp, level, streak }) {
  const xpForNextLevel = 100;
  const progressPercent = (xp % xpForNextLevel) / xpForNextLevel * 100;

  return (
    <div className={styles.gamification}>
      <div>Level: {level}</div>
      <div className={styles.xpBar}>
        <div
          className={styles.xpProgress}
          style={{ width: `${progressPercent}%` }}
        />
      </div>
      <div>XP: {xp} / {xpForNextLevel}</div>
      <div>🔥 Streak: {streak} day{streak !== 1 ? 's' : ''}</div>
    </div>
  );
}
