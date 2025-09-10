import React, { useEffect, useRef } from "react";

export default function MiniGame({ onScore }) {
  const canvasRef = useRef(null);
  // Use a ref for the animation frame to cancel it on unmount
  const animationFrameId = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = (canvas.width = 600);
    const height = (canvas.height = 150);

    // --- Game State Variables ---
    // These are now regular variables within the useEffect scope,
    // which prevents them from triggering re-renders.
    let plantY = height - 30;
    let plantDY = 0;
    const gravity = 0.8;
    const jumpPower = -12;
    let obstacles = [];
    let frame = 0;
    let score = 0;
    let gameOver = false;

    function gameLoop() {
      if (gameOver) return; // Stop the loop if the game has ended

      ctx.clearRect(0, 0, width, height);
      ctx.font = "24px sans-serif"; // Set font for emojis

      // Ground
      ctx.fillStyle = "#A1887F"; // A more 'earthy' color
      ctx.fillRect(0, height - 20, width, 20);

      // Player (Leaf 🍂)
      plantDY += gravity;
      plantY += plantDY;

      // Prevent falling through the ground
      if (plantY > height - 30) {
        plantY = height - 30;
        plantDY = 0;
      }
      // Note: The y-coordinate for fillText is the baseline, not the top corner.
      ctx.fillText("🍂", 50, plantY);

      // Obstacles (Tumbleweed 🌵)
      // Spawn a new obstacle every 90 frames
      if (frame % 90 === 0) {
        obstacles.push({ x: width, w: 20, h: 20 });
      }

      obstacles.forEach((obs, i) => {
        obs.x -= 6; // Move obstacle to the left
        // The obstacle's y-position is fixed to the ground
        const obsY = height - 30;
        ctx.fillText("🌵", obs.x, obsY);

        // --- Collision Detection ---
        // Using simple Axis-Aligned Bounding Box (AABB) collision.
        // Player Box: x=50, y=plantY-20, w=20, h=20
        // Obstacle Box: x=obs.x, y=obsY-obs.h, w=obs.w, h=obs.h
        const player = { x: 50, y: plantY - 20, w: 20, h: 20 };

        if (
          player.x < obs.x + obs.w &&
          player.x + player.w > obs.x &&
          player.y < obsY &&
          player.y + player.h > obsY - obs.h
        ) {
          gameOver = true; // Set the local gameOver flag
          ctx.fillStyle = "black";
          ctx.textAlign = "center";
          ctx.fillText(
            "Game Over! Press 'R' to restart",
            width / 2,
            height / 2
          );
          if (onScore) onScore(Math.floor(score));
          // Don't continue the loop
          return;
        }

        // Remove obstacles that have gone off-screen
        if (obs.x + obs.w < 0) {
          obstacles.splice(i, 1);
        }
      });

      // Update and draw score
      score += 0.1;
      ctx.fillStyle = "black";
      ctx.textAlign = "start"; // Reset text alignment
      ctx.font = "16px Arial";
      ctx.fillText("Score: " + Math.floor(score), width - 100, 30);

      frame++;
      animationFrameId.current = requestAnimationFrame(gameLoop);
    }

    function restartGame() {
      // Reset all game variables to their initial state
      gameOver = false;
      score = 0;
      obstacles = [];
      plantY = height - 30;
      plantDY = 0;
      frame = 0;
      // Start the game loop again
      gameLoop();
    }

    function handleKey(e) {
      // Jump only if the leaf is on the ground
      if (e.code === "Space" && plantY === height - 30) {
        plantDY = jumpPower;
      }
      // Restart the game if it's over
      if (e.code === "KeyR" && gameOver) {
        restartGame();
      }
    }

    // Start the game
    gameLoop();

    // Add event listeners
    window.addEventListener("keydown", handleKey);

    // Cleanup function to remove the listener and cancel animation frame
    return () => {
      window.removeEventListener("keydown", handleKey);
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
    // An empty dependency array ensures this effect runs only once on mount.
  }, [onScore]);

  return (
    <canvas
      ref={canvasRef}
      width={600}
      height={150}
      className="border border-gray-400 rounded-lg mt-4 shadow-md"
    />
  );
}