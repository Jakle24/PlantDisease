import React, { useEffect, useRef, useState } from "react";

export default function MiniGame({ onScore }) {
  const canvasRef = useRef(null);
  const [score, setScore] = useState(0);
  const [gameOver, setGameOver] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const width = canvas.width = 600;
    const height = canvas.height = 150;

    let plantY = height - 30;
    let plantDY = 0;
    const gravity = 0.8;
    const jumpPower = -12;
    const obstacles = [];
    let frame = 0;

    function gameLoop() {
      ctx.clearRect(0, 0, width, height);

      // Ground
      ctx.fillStyle = "#6B8E23";
      ctx.fillRect(0, height - 20, width, 20);

      // Plant
      plantDY += gravity;
      plantY += plantDY;
      if (plantY > height - 30) { plantY = height - 30; plantDY = 0; }
      ctx.fillStyle = "#228B22";
      ctx.fillRect(50, plantY - 20, 20, 20);

      // Obstacles
      if (frame % 90 === 0) obstacles.push({ x: width, y: height - 30, w: 20, h: 20 });
      obstacles.forEach((obs, i) => {
        obs.x -= 6;
        ctx.fillStyle = "#A52A2A";
        ctx.fillRect(obs.x, obs.y - obs.h, obs.w, obs.h);

        // Collision
        if (50 < obs.x + obs.w && 50 + 20 > obs.x && plantY - 20 < obs.y && plantY > obs.y - obs.h) {
          setGameOver(true);
        }

        if (obs.x + obs.w < 0) obstacles.splice(i, 1);
      });

      if (!gameOver) setScore(prev => prev + 0.1);
      ctx.fillStyle = "#000";
      ctx.font = "16px Arial";
      ctx.fillText("Score: " + Math.floor(score), width - 100, 30);

      frame++;
      if (!gameOver) requestAnimationFrame(gameLoop);
      else {
        ctx.fillText("Game Over! Press R to restart", width / 2 - 100, height / 2);
        if (onScore) onScore(Math.floor(score));
      }
    }

    gameLoop();

    function handleKey(e) {
      if (e.code === "Space" && plantY === height - 30) plantDY = jumpPower;
      if (e.code === "KeyR" && gameOver) {
        setScore(0);
        setGameOver(false);
        obstacles.length = 0;
        plantY = height - 30;
        plantDY = 0;
        requestAnimationFrame(gameLoop);
      }
    }

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [gameOver, onScore]);

  return (
    <canvas ref={canvasRef} width={600} height={150} className="border border-gray-400 rounded mt-4"/>
  );
}
