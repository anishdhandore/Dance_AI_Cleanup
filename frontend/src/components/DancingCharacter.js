import React, { useRef, useEffect, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGLTF, useAnimations, OrbitControls } from '@react-three/drei';

// Map emotions to animation file names
const emotionToFile = {
  'Positive_Affect_Joy': 'happy.glb',
  'Sadness_Low_Arousal_Negative': 'sad.glb',
  'Anger_High_Arousal_Negative': 'anger.glb',
  'Fear_Anxiety': 'fear.glb',
  'Surprise_Epistemic': 'surprise.glb',
  'Neutral_Cat': 'normal.glb', // Neutral = no dance
};

// Priority order for emotions (higher index = higher priority)
const emotionPriority = {
  'Anger_High_Arousal_Negative': 5,
  'Fear_Anxiety': 4,
  'Surprise_Epistemic': 3,
  'Positive_Affect_Joy': 2,
  'Sadness_Low_Arousal_Negative': 1,
  'Neutral_Cat': 0
};

// Function to select the best emotion for animation
function selectBestEmotion(emotions) {
  if (!emotions || emotions.length === 0) {
    return 'Neutral_Cat';
  }
  
  // If only one emotion, use it
  if (emotions.length === 1) {
    return emotions[0];
  }
  
  // If multiple emotions, select the one with highest priority
  let bestEmotion = emotions[0];
  let highestPriority = emotionPriority[emotions[0]] || 0;
  
  for (const emotion of emotions) {
    const priority = emotionPriority[emotion] || 0;
    if (priority > highestPriority) {
      highestPriority = priority;
      bestEmotion = emotion;
    }
  }
  
  console.log(`Multiple emotions detected: ${emotions}. Selected: ${bestEmotion} (priority: ${highestPriority})`);
  return bestEmotion;
}

// Dance animation component
function DanceAnimation({ file, onFinish, intensity }) {
  const group = useRef();
  const { scene, animations } = useGLTF(`/models/animations/${file}`);
  const { actions, mixer } = useAnimations(animations, group);
  const [action, setAction] = useState(null);

  useEffect(() => {
    if (!actions) return;
    const actionNames = Object.keys(actions);
    if (actionNames.length === 0) return;
    const anim = actions[actionNames[0]];
    if (action !== anim) {
      if (action) action.fadeOut(0.2);

      // Set speed based on intensity, defaulting if not provided.
      // Speed will range from 0.4 (intensity=0) to 1.0 (intensity=1).
      const speed = 0.4 + (intensity || 0.5) * 0.6;
      anim.timeScale = speed;

      anim.reset().fadeIn(0.2).play();
      setAction(anim);
      const duration = anim.getClip().duration || 4;
      const timeout = setTimeout(() => {
        if (onFinish) onFinish();
      }, duration * 1000);
      return () => clearTimeout(timeout);
    }
  }, [actions, action, onFinish, intensity]);

  useFrame((_, delta) => {
    if (mixer) mixer.update(delta);
  });

  return (
    <group ref={group} dispose={null} position={[0, -3.5, 0]} scale={[0.68, 0.68, 0.68]}>
      <primitive object={scene} />
    </group>
  );
}

// Main character component
function DancingCharacter({ emotions, intensity, triggerDance }) {
  const idleGroup = useRef();
  const [idleLoaded, setIdleLoaded] = useState(false);
  const [idleError, setIdleError] = useState(null);
  const [idleAction, setIdleAction] = useState(null);

  const [danceFile, setDanceFile] = useState(null);
  const [showDance, setShowDance] = useState(false);

  const { scene: idleScene, animations: idleAnimations } = useGLTF('/models/idle_character.glb', 
    undefined,
    (error) => setIdleError(error)
  );
  const { actions: idleActions, mixer: idleMixer } = useAnimations(idleAnimations, idleGroup);

  // Trigger dance when triggerDance value changes
  useEffect(() => {
    console.log("Dance trigger received. Emotions:", emotions);
    if (emotions && emotions.length > 0) {
      const primaryEmotion = selectBestEmotion(emotions);
      const file = emotionToFile[primaryEmotion];
      console.log("Mapped to file:", file);
      if (file) {
        setDanceFile(file);
        setShowDance(true);
      }
    } else {
      console.warn("No emotions provided when dance was triggered.");
    }
  }, [triggerDance, emotions]);

  useEffect(() => {
    if (idleScene) {
      idleScene.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });
      setIdleLoaded(true);
    }
  }, [idleScene]);

  useEffect(() => {
    if (!idleLoaded || !idleActions) return;
    const actionNames = Object.keys(idleActions);
    if (actionNames.length === 0) return;
    const action =
      idleActions['idle'] ||
      idleActions['Idle'] ||
      idleActions['normal'] ||
      idleActions['Normal'] ||
      idleActions[actionNames[0]];
    if (idleAction !== action) {
      if (idleAction) idleAction.fadeOut(0.5);
      action.reset().fadeIn(0.5).play();
      setIdleAction(action);
    }
  }, [idleLoaded, idleActions, idleAction]);

  useFrame((_, delta) => {
    if (idleMixer) idleMixer.update(delta);
  });

  if (idleError) {
    return (
      <mesh>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="red" />
      </mesh>
    );
  }

  if (!idleLoaded) {
    return (
      <mesh>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="gray" />
      </mesh>
    );
  }

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight
        position={[5, 5, 5]}
        intensity={1.2}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <directionalLight
        position={[-5, 5, -5]}
        intensity={0.8}
      />
      <hemisphereLight
        intensity={0.5}
        groundColor={[0.1, 0.1, 0.1]}
        skyColor={[0.8, 0.8, 1]}
      />
      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        minDistance={3}
        maxDistance={12}
        initialPosition={[0, -1, 5]}
        target={[0, -2, 0]}
      />
      {!showDance && (
        <group ref={idleGroup} dispose={null} position={[0, -3.5, 0]} scale={[0.68, 0.68, 0.68]}>
          <primitive object={idleScene} />
        </group>
      )}
      {showDance && danceFile && (
        <DanceAnimation
          key={danceFile}
          file={danceFile}
          intensity={intensity}
          onFinish={() => {
            setShowDance(false);
            setDanceFile(null);
          }}
        />
      )}
    </>
  );
}

export default DancingCharacter;
