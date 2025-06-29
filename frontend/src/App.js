import React, { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import DancingCharacter from './components/DancingCharacter';
import './App.css';

const Loader = () => <div className="loader"></div>;

function App() {
  const [inputText, setInputText] = useState('');
  const [emotions, setEmotions] = useState([]);
  const [intensity, setIntensity] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [danceTrigger, setDanceTrigger] = useState(0);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });
      
      const data = await response.json();
      setEmotions(data.emotions || []);
      setIntensity(data.intensity || 0);
      setDanceTrigger(t => t + 1); // Trigger the dance animation
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setEmotions([]);
      setIntensity(0);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="character-canvas">
        <Canvas camera={{ position: [0, -1, 5], fov: 50 }}>
          <ambientLight intensity={0.8} />
          <directionalLight 
            position={[5, 10, 7]} 
            intensity={1.5} 
            castShadow 
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
          />
          <DancingCharacter 
            emotions={emotions} 
            intensity={intensity} 
            triggerDance={danceTrigger} 
          />
          {/* Simple floor */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -3.5, 0]} receiveShadow>
            <planeGeometry args={[50, 50]} />
            <meshStandardMaterial color="#0a0a0a" roughness={0.7} />
          </mesh>
          <OrbitControls 
            minDistance={3} 
            maxDistance={12} 
            enablePan={false}
            target={[0, -2, 0]}
          />
        </Canvas>
        
        {/* Floating Orbs */}
        <div className="floating-orbs">
          <div className="orb"></div>
          <div className="orb"></div>
          <div className="orb"></div>
          <div className="orb"></div>
          <div className="orb"></div>
          <div className="orb"></div>
          <div className="orb"></div>
          <div className="orb"></div>
          <div className="orb"></div>
        </div>
      </div>

      <div className="ui-panel">
        <h1 className="app-title">AI Dance Generator</h1>
        <form onSubmit={handleSubmit} className="dance-form">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="How are you feeling today?"
            className="text-input"
            disabled={isLoading}
          />
          <button type="submit" className="submit-button" disabled={isLoading || !inputText.trim()}>
            {isLoading ? <Loader /> : 'Generate'}
          </button>
        </form>
        
        {emotions.length > 0 && !isLoading && (
          <div className="predictions-container">
            <div className="emotions-list">
              {emotions.map((emotion, index) => (
                <span key={index} className="emotion-tag">
                  {emotion.replace(/_/g, ' ')}
                  {index === 0 && emotions.length > 1 && <span className="primary-indicator"> (Primary)</span>}
                </span>
              ))}
            </div>
            <p className="intensity-display">
              Intensity: {typeof intensity === 'number' ? intensity.toFixed(2) : 'N/A'}
            </p>
          </div>
        )}
      </div>

      {/* Floating UI for Character Area */}
      {emotions.length > 0 && !isLoading && (
        <div className="floating-ui">
          <div style={{ marginBottom: '0.5rem', fontWeight: '600' }}>
            🎭 {emotions[0]?.replace(/_/g, ' ')}
          </div>
          <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>
            Intensity: {typeof intensity === 'number' ? intensity.toFixed(2) : 'N/A'}
          </div>
        </div>
      )}

      {/* Badge in top-right corner */}
      <div className="badge-container">
        <a 
          href="https://bolt.new/" 
          target="_blank" 
          rel="noopener noreferrer"
          className="badge-link"
          aria-label="Powered by Bolt"
        >
          <img 
            src="/images/white_circle_360x360.png" 
            alt="Bolt Badge" 
            className="badge-image"
          />
        </a>
      </div>
    </div>
  );
}

export default App; 