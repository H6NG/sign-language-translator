import { useState, useEffect, useRef, useCallback } from 'react'
import { useBackendStatus } from './useBackendStatus'
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import HomePage from './HomePage'
import Settings from './Settings'
import './App.css'

import EnhancedPage from './EnhancedPage'
import Transcriber from './Transcriber'
import GuidePage from './GuidePage'
import HistoryPage from './HistoryPage'
import QuizPage from './QuizPage'
import PracticePage from './PracticePage'
import PrivacyPolicy from './PrivacyPolicy'
import TermsOfService from './TermsOfService'
import CookiePolicy from './CookiePolicy'

const API_URL = 'http://localhost:5001'

function TrackerPage({ theme, onSettingsOpen }) {
  const navigate = useNavigate()
  const [status, setStatus] = useState({
    buffer_level: 0,
    buffer_size: 200,
    capture_count: 0,
    is_running: true,
    has_hands: false,
    model_loaded: false
  })
  const [prediction, setPrediction] = useState({
    letter: '',
    confidence: 0,
    mode: 'both',
    top3: [],
    word: '',
    word_confidence: 0,
    hand_position: null
  })
  const [landmarks, setLandmarks] = useState(null)
  const [handFeedback, setHandFeedback] = useState({ status: 'ok', hint: '' })
  const [isCapturing, setIsCapturing] = useState(false)
  const [landmarkHistory, setLandmarkHistory] = useState([])
  const [arOverlayEnabled, setArOverlayEnabled] = useState(true)
  const canvasRef = useRef(null)
  const videoContainerRef = useRef(null)

  // Sentence Builder state
  const [sentence, setSentence] = useState('')
  const [lastAddedLetter, setLastAddedLetter] = useState('')

  // Fullscreen state
  const [isFullscreen, setIsFullscreen] = useState(false)

  // Prediction history for confusion matrix
  const [predictionHistory, setPredictionHistory] = useState([])

  // Text-to-Speech state
  const [isSpeaking, setIsSpeaking] = useState(false)

  // Auto-add tracking
  const stableLetterRef = useRef('')
  const stableTimerRef = useRef(null)

  // Configurable thresholds (from localStorage, set via Settings)
  const getAutoAddThreshold = () => parseFloat(localStorage.getItem('slt_auto_add_threshold') || '0.5')
  const getSoundEnabled = () => localStorage.getItem('slt_sound_on_add') !== 'false'

  const playAddBeep = useCallback(() => {
    if (!getSoundEnabled()) return
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)()
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      osc.connect(gain)
      gain.connect(ctx.destination)
      osc.frequency.value = 400
      osc.type = 'sine'
      gain.gain.setValueAtTime(0.15, ctx.currentTime)
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.05)
      osc.start(ctx.currentTime)
      osc.stop(ctx.currentTime + 0.05)
    } catch { /* ignore */ }
  }, [])
  const getManualAddThreshold = () => parseFloat(localStorage.getItem('slt_manual_add_threshold') || '0.3')
  const getAutoAddDelay = () => parseFloat(localStorage.getItem('slt_auto_add_delay') || '2') * 1000

  // Word autocomplete
  const [wordSuggestions, setWordSuggestions] = useState([])
  const wordDictRef = useRef(null)
  const loadingDictRef = useRef(false)

  // Fetch status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_URL}/status`)
        const data = await res.json()
        setStatus(data)
      } catch (err) {
        setStatus(prev => ({ ...prev, is_running: false }))
      }
    }

    fetchStatus()
    const interval = setInterval(fetchStatus, 500)
    return () => clearInterval(interval)
  }, [])

  // Fetch prediction periodically
  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        const res = await fetch(`${API_URL}/prediction`)
        const data = await res.json()
        setPrediction(data)
      } catch (err) {
        // Silent fail
      }
    }

    const interval = setInterval(fetchPrediction, 100) // 10fps
    return () => clearInterval(interval)
  }, [])

  // Auto-add letter after stable prediction (configurable threshold & delay)
  useEffect(() => {
    const currentLetter = prediction.letter
    const confidence = prediction.confidence
    const autoThreshold = getAutoAddThreshold()

    // Only consider predictions above the auto-add threshold
    if (!currentLetter || confidence < autoThreshold) {
      // Clear timer if no valid prediction
      if (stableTimerRef.current) {
        clearTimeout(stableTimerRef.current)
        stableTimerRef.current = null
      }
      stableLetterRef.current = ''
      return
    }

    // If same letter, don't restart timer
    if (currentLetter === stableLetterRef.current) {
      return // Keep existing timer running
    }

    // New letter - reset and start timer
    stableLetterRef.current = currentLetter

    // Clear existing timer
    if (stableTimerRef.current) {
      clearTimeout(stableTimerRef.current)
    }

    // Start new timer with configurable delay
    const top3Snap = prediction.top3 || []
    stableTimerRef.current = setTimeout(() => {
      const letterToAdd = String(currentLetter)
      playAddBeep()
      setSentence(prev => prev + letterToAdd)
      setLastAddedLetter(letterToAdd)
      setPredictionHistory(prev => [...prev.slice(-99), {
        letter: letterToAdd,
        confidence: confidence,
        timestamp: Date.now(),
        top3: top3Snap
      }])
      stableLetterRef.current = ''
      stableTimerRef.current = null
    }, getAutoAddDelay())

    // Don't clear timer in cleanup - we manage it manually
  }, [prediction.letter, prediction.confidence, getAutoAddThreshold, getAutoAddDelay, playAddBeep])

  // Auto-add word for gesture predictions
  useEffect(() => {
    const currentGesture = prediction.gesture
    const confidence = prediction.gesture_confidence
    const autoThreshold = getAutoAddThreshold()

    if (!currentGesture || confidence < autoThreshold) {
      return
    }

    // Use the same stable timer mechanism, but for gesture words
    if (currentGesture === stableLetterRef.current) {
      return
    }

    stableLetterRef.current = currentGesture

    if (stableTimerRef.current) {
      clearTimeout(stableTimerRef.current)
    }

    const top3Snap = prediction.gesture_top3 || []
    stableTimerRef.current = setTimeout(() => {
      const wordToAdd = String(currentGesture)
      playAddBeep()
      // Add a space before words if there's existing text but no trailing space
      setSentence(prev => {
        const prefix = (prev.length > 0 && !prev.endsWith(' ')) ? prev + ' ' : prev
        return prefix + wordToAdd.toUpperCase() + ' '
      })
      setLastAddedLetter(wordToAdd.toUpperCase())
      setPredictionHistory(prev => [...prev.slice(-99), {
        letter: wordToAdd.toUpperCase(),
        confidence: confidence,
        timestamp: Date.now(),
        top3: top3Snap
      }])
      stableLetterRef.current = ''
      stableTimerRef.current = null
    }, getAutoAddDelay())

  }, [prediction.gesture, prediction.gesture_confidence, getAutoAddThreshold, getAutoAddDelay, playAddBeep])

  // Word autocomplete - lazy-load dictionary and compute suggestions
  useEffect(() => {
    const words = sentence.split(' ')
    const currentWord = words[words.length - 1]?.toLowerCase() || ''

    if (currentWord.length < 2) {
      setWordSuggestions([])
      return
    }

    const computeSuggestions = (dict) => {
      const prefix = currentWord
      const matches = []
      for (const word of Object.keys(dict)) {
        if (word.startsWith(prefix) && word !== prefix && word.length >= 3 && word.length <= 12) {
          matches.push(word)
          if (matches.length >= 5) break
        }
      }
      setWordSuggestions(matches)
    }

    if (wordDictRef.current) {
      computeSuggestions(wordDictRef.current)
    } else if (!loadingDictRef.current) {
      loadingDictRef.current = true
      fetch('/words_dictionary.json')
        .then(res => res.json())
        .then(data => {
          wordDictRef.current = data
          computeSuggestions(data)
        })
        .catch(err => console.error('Failed to load dictionary:', err))
    }
  }, [sentence])

  // Accept a word suggestion
  const acceptSuggestion = useCallback((word) => {
    setSentence(prev => {
      const words = prev.split(' ')
      words[words.length - 1] = word.toUpperCase()
      return words.join(' ')
    })
    setWordSuggestions([])
  }, [])

  // Fetch hand position feedback
  useEffect(() => {
    const fetchFeedback = async () => {
      try {
        const res = await fetch(`${API_URL}/feedback`)
        const data = await res.json()
        if (data.status && data.status !== 'ok') {
          setHandFeedback({ status: data.status, hint: data.hint || '' })
        } else {
          setHandFeedback({ status: 'ok', hint: '' })
        }
      } catch { /* silent */ }
    }
    const interval = setInterval(fetchFeedback, 500)
    return () => clearInterval(interval)
  }, [])

  // Fetch landmarks for graph
  useEffect(() => {
    const fetchLandmarks = async () => {
      try {
        const res = await fetch(`${API_URL}/landmarks`)
        const data = await res.json()
        if (data.has_data) {
          setLandmarks(data.fingertips)
          setLandmarkHistory(prev => {
            const newHistory = [...prev, data.fingertips]
            return newHistory.slice(-50) // Keep last 50 frames
          })
        }
      } catch (err) {
        // Silent fail
      }
    }

    const interval = setInterval(fetchLandmarks, 100) // 10fps
    return () => clearInterval(interval)
  }, [])

  // Draw landmark graph
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || landmarkHistory.length < 2) return

    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height

    // Clear - use theme-aware colors
    ctx.fillStyle = theme === 'light' ? '#f1f5f9' : '#12121a'
    ctx.fillRect(0, 0, width, height)

    // Draw grid
    ctx.strokeStyle = theme === 'light' ? '#cbd5e1' : '#2a2a3a'
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const y = (height / 4) * i
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    const colors = ['#ef4444', '#f59e0b', '#10b981', '#6366f1', '#8b5cf6']

    // Draw lines for each finger
    fingers.forEach((finger, fingerIdx) => {
      ctx.strokeStyle = colors[fingerIdx]
      ctx.lineWidth = 2
      ctx.beginPath()

      landmarkHistory.forEach((frame, i) => {
        const x = (width / (landmarkHistory.length - 1)) * i
        const y = height - (frame[finger]?.y || 0.5) * height

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.stroke()
    })
  }, [landmarkHistory, theme])

  // Handle capture
  const handleCapture = useCallback(async () => {
    if (isCapturing || status.buffer_level < status.buffer_size) return

    setIsCapturing(true)
    try {
      const res = await fetch(`${API_URL}/capture`, { method: 'POST' })
      const data = await res.json()
      if (data.success) {
        setStatus(prev => ({ ...prev, capture_count: data.capture_count }))
      }
    } catch (err) {
      console.error('Capture failed:', err)
    }
    setIsCapturing(false)
  }, [isCapturing, status.buffer_level, status.buffer_size])

  // Handle quit
  const handleQuit = async () => {
    if (window.confirm('Are you sure you want to quit?')) {
      try {
        await fetch(`${API_URL}/quit`, { method: 'POST' })
      } catch (err) {
        console.error('Quit failed:', err)
      }
    }
  }

  // Handle mode change
  const handleModeChange = async (mode) => {
    try {
      await fetch(`${API_URL}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prediction_mode: mode })
      })
      setPrediction(prev => ({ ...prev, mode }))
    } catch (err) {
      console.error('Mode change failed:', err)
    }
  }

  // Fullscreen toggle function (defined before useEffect that uses it)
  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      videoContainerRef.current?.requestFullscreen()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen()
      setIsFullscreen(false)
    }
  }, [])

  // Copy sentence to clipboard
  const copySentence = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(sentence)
    } catch (err) {
      console.error('Copy failed:', err)
    }
  }, [sentence])

  // Text-to-Speech
  const utteranceRef = useRef(null)

  const speakSentence = useCallback(() => {
    if (!sentence) return

    // Stop any current speech first
    speechSynthesis.cancel()

    // Space out individual characters so TTS pronounces each one clearly
    // e.g. "HELLO" becomes "H. E. L. L. O." which TTS reads letter by letter
    const textToSpeak = sentence.includes(' ')
      ? sentence
      : sentence.split('').join('. ') + '.'

    // Small delay after cancel() to avoid Chrome race condition
    setTimeout(() => {
      const utterance = new SpeechSynthesisUtterance(textToSpeak)
      utterance.rate = 0.85
      utterance.pitch = 1.0
      utterance.onstart = () => setIsSpeaking(true)
      utterance.onend = () => setIsSpeaking(false)
      utterance.onerror = (e) => {
        // 'interrupted' is expected when user clicks Stop
        if (e.error !== 'interrupted') {
          console.error('TTS error:', e.error)
        }
        setIsSpeaking(false)
      }

      // Keep a ref to prevent garbage collection (Chrome bug)
      utteranceRef.current = utterance
      speechSynthesis.speak(utterance)
    }, 50)
  }, [sentence])

  const stopSpeaking = useCallback(() => {
    speechSynthesis.cancel()
    utteranceRef.current = null
    setIsSpeaking(false)
  }, [])

  // Save session to localStorage
  const saveSession = useCallback(() => {
    if (!sentence && predictionHistory.length === 0) return
    const sessions = JSON.parse(localStorage.getItem('slt_sessions') || '[]')
    const newSession = {
      id: Date.now(),
      timestamp: new Date().toISOString(),
      sentence,
      predictionHistory: predictionHistory.slice(),
      letterCount: sentence.replace(/\s/g, '').length,
      avgConfidence: predictionHistory.length > 0
        ? predictionHistory.reduce((sum, p) => sum + p.confidence, 0) / predictionHistory.length
        : 0
    }
    sessions.unshift(newSession)
    localStorage.setItem('slt_sessions', JSON.stringify(sessions))
    return true
  }, [sentence, predictionHistory])

  // Add letter to sentence helper (uses configurable manual-add threshold)
  const addLetterToSentence = useCallback(() => {
    const manualThreshold = getManualAddThreshold()

    // Check if we have a gesture prediction first
    if (prediction.mode === 'gestures' && prediction.gesture && prediction.gesture_confidence > manualThreshold) {
      playAddBeep()
      const wordToAdd = String(prediction.gesture).toUpperCase()
      setSentence(prev => {
        const prefix = (prev.length > 0 && !prev.endsWith(' ')) ? prev + ' ' : prev
        return prefix + wordToAdd + ' '
      })
      setLastAddedLetter(wordToAdd)
      setPredictionHistory(prev => [...prev.slice(-99), {
        letter: wordToAdd,
        confidence: prediction.gesture_confidence,
        timestamp: Date.now(),
        top3: prediction.gesture_top3 || []
      }])
      return
    }

    // Otherwise fallback to letter prediction
    if (prediction.letter && prediction.confidence > manualThreshold) {
      playAddBeep()
      const letterToAdd = String(prediction.letter)
      setSentence(prev => {
        const newSentence = prev + letterToAdd
        return newSentence
      })
      setLastAddedLetter(letterToAdd)
      setPredictionHistory(prev => [...prev.slice(-99), {
        letter: letterToAdd,
        confidence: prediction.confidence,
        timestamp: Date.now(),
        top3: prediction.top3 || []
      }])
    }
  }, [prediction, getManualAddThreshold, playAddBeep])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Don't trigger shortcuts if user is typing in an input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

      switch (e.key.toLowerCase()) {
        case 'c':
          // Capture gesture
          if (status.buffer_level >= status.buffer_size && !isCapturing) {
            handleCapture()
          }
          break
        case 'q':
          // Quit application
          handleQuit()
          break
        case '1':
          // Switch to letters mode
          handleModeChange('letters')
          break
        case '2':
          // Switch to numbers mode
          handleModeChange('numbers')
          break
        case '3':
          // Switch to both mode
          handleModeChange('both')
          break
        case 'a':
          // Toggle AR overlay
          setArOverlayEnabled(prev => !prev)
          break
        case ' ':
          // Add space to sentence
          e.preventDefault()
          setSentence(prev => prev + ' ')
          break
        case 'enter':
          // Add current prediction to sentence
          e.preventDefault()
          addLetterToSentence()
          break
        case 'backspace':
          // Remove last character from sentence
          e.preventDefault()
          setSentence(prev => prev.slice(0, -1))
          break
        case 'x':
          // Clear sentence
          setSentence('')
          setLastAddedLetter('')
          break
        case 'f':
          // Toggle fullscreen
          toggleFullscreen()
          break
        default:
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [status.buffer_level, status.buffer_size, isCapturing, handleCapture, prediction, toggleFullscreen, addLetterToSentence])

  // Listen for fullscreen change
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  const bufferPercent = (status.buffer_level / status.buffer_size) * 100
  const isBufferFull = status.buffer_level >= status.buffer_size

  return (
    <div className="tracker-page">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <button className="back-btn" onClick={() => navigate('/')}>
            ← Back
          </button>
          <h1>🤟 Sign Language Translator</h1>
        </div>
        <div className="header-status">
          <div className="status-indicator">
            <span className={`status-dot ${status.is_running ? '' : 'offline'}`}></span>
            {status.is_running ? 'Connected' : 'Disconnected'}
          </div>
          {status.model_loaded && (
            <div className="status-indicator ai-badge">
              <span>🧠</span> AI Active
            </div>
          )}
          <button className="settings-btn" onClick={onSettingsOpen}>
            ⚙️
          </button>
        </div>
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Video Section */}
        <div className="video-section">
          <div className={`video-container ${isFullscreen ? 'fullscreen' : ''}`} ref={videoContainerRef}>
            <img
              className="video-feed"
              src={`${API_URL}/video_feed`}
              alt="Hand Tracking Feed"
            />
            <div className="video-overlay">
              <span className="overlay-badge">LIVE</span>
              {status.has_hands && (
                <span className="overlay-badge recording">TRACKING</span>
              )}
            </div>
            {handFeedback.status !== 'ok' && handFeedback.hint && (
              <div className="feedback-toast">
                <span className="feedback-icon">⚠️</span>
                <span>{handFeedback.hint}</span>
              </div>
            )}

            {/* Fullscreen button - positioned bottom right of video */}
            <button className="fullscreen-btn" onClick={toggleFullscreen}>
              {isFullscreen ? '⛶' : '⛶'}
            </button>

            {/* AR Prediction Overlay - Moves with hand, positioned to side */}
            {arOverlayEnabled && (() => {
              const mode = prediction.mode;
              const isGesture = mode === 'gestures';
              const predText = isGesture ? prediction.gesture : prediction.letter;
              const predConf = isGesture ? prediction.gesture_confidence : prediction.confidence;

              if (!predText || predConf <= 0.3) return null;

              const hp = prediction.hand_position
              if (!hp) return (
                <div className="ar-prediction-overlay">
                  <div className={`ar-letter ${isGesture ? 'gesture' : ''}`}>{isGesture ? predText.toUpperCase() : predText}</div>
                  <div className="ar-confidence">
                    {(predConf * 100).toFixed(0)}%
                  </div>
                </div>
              )

              // Position overlay to the LEFT of hand if hand is on right side, and vice versa
              const handOnRightSide = hp.x > 0.5
              const xOffset = handOnRightSide ? -12 : 12 // percentage offset from hand (closer)
              const xPos = Math.min(Math.max((hp.x * 100) + xOffset, 8), 92)
              const yPos = Math.min(Math.max(hp.y * 100, 10), 85)

              return (
                <div
                  className="ar-prediction-overlay"
                  style={{
                    left: `${xPos}%`,
                    top: `${yPos}%`,
                    transform: handOnRightSide ? 'translate(-100%, -50%)' : 'translate(0%, -50%)',
                    right: 'auto'
                  }}
                >
                  <div className={`ar-letter ${isGesture ? 'gesture' : ''}`}>{isGesture ? predText.toUpperCase() : predText}</div>
                  <div className="ar-confidence">
                    {(predConf * 100).toFixed(0)}%
                  </div>
                </div>
              )
            })()}
          </div>

          {/* Control Panel */}
          <div className="control-panel">
            <button
              className="btn btn-capture"
              onClick={handleCapture}
              disabled={!isBufferFull || isCapturing}
            >
              {isCapturing ? '⏳ Saving...' : '📸 Capture Gesture'}
            </button>
            <button className="btn btn-quit" onClick={handleQuit}>
              ⏹️ Quit
            </button>
          </div>
        </div>

        {/* Sidebar */}
        <div className="sidebar">
          {/* Prediction Card */}
          <div className="card prediction-card">
            <div className="card-header">
              <span className="card-title">🧠 AI Prediction</span>
              <span className={`card-badge ${status.model_loaded ? 'active' : ''}`}>
                {status.model_loaded ? 'ACTIVE' : 'LOADING'}
              </span>
            </div>

            <div className="prediction-display">
              {(() => {
                const isGesture = prediction.mode === 'gestures';
                const mainText = isGesture ? prediction.gesture : prediction.letter;
                const top3 = isGesture ? prediction.gesture_top3 : prediction.top3;

                if (top3 && top3.length > 0) {
                  return (
                    <div className="top3-predictions">
                      {top3.map((pred, index) => (
                        <div key={`${pred.letter}-${index}`} className={`top3-item ${index === 0 ? 'primary' : ''}`}>
                          <div className="top3-rank">#{index + 1}</div>
                          <div className={`top3-letter ${isGesture ? 'gesture-text' : ''}`}>{isGesture ? String(pred.letter).toUpperCase() : pred.letter}</div>
                          <div className="top3-bar-container">
                            <div
                              className="top3-bar-fill"
                              style={{ width: `${pred.confidence * 100}%` }}
                            ></div>
                          </div>
                          <div className="top3-confidence">
                            {(pred.confidence * 100).toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  );
                } else {
                  return (
                    <div className="no-prediction">
                      {status.has_hands ? 'Analyzing...' : 'Show a sign'}
                    </div>
                  );
                }
              })()}
            </div>

            {/* Word prediction (LSTM) */}
            {prediction.word && prediction.word_confidence > 0.5 && (
              <div className="word-prediction-bar">
                <span className="word-label">Word:</span>
                <span className="word-value">{prediction.word}</span>
                <span className="word-conf">{(prediction.word_confidence * 100).toFixed(0)}%</span>
                <button
                  className="btn btn-small btn-primary"
                  onClick={() => {
                    setSentence(prev => (prev ? prev + ' ' : '') + prediction.word)
                    setLastAddedLetter('')
                  }}
                >
                  + Add Word
                </button>
              </div>
            )}

            {/* Mode Selector */}
            <div className="mode-selector">
              <span className="mode-label">Mode:</span>
              <div className="mode-buttons-grid">
                {['letters', 'numbers', 'both', 'gestures'].map(mode => (
                  <button
                    key={mode}
                    className={`mode-btn ${prediction.mode === mode ? 'active' : ''}`}
                    onClick={() => handleModeChange(mode)}
                  >
                    {mode === 'letters' ? 'A-Z' : mode === 'numbers' ? '0-9' : mode === 'gestures' ? '👋 Gestures' : 'All'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Sentence Builder - moved to second position */}
          <div className="card sentence-builder-card">
            <div className="card-header">
              <span className="card-title">📝 Sentence Builder</span>
              <span className="card-badge">{sentence.length} chars</span>
            </div>

            <div className="sentence-display">
              {sentence || <span className="placeholder">Press Enter to add letters...</span>}
              {lastAddedLetter && <span className="last-added">+{lastAddedLetter}</span>}
            </div>

            {/* Word Autocomplete Suggestions */}
            {wordSuggestions.length > 0 && (
              <div className="word-suggestions">
                <span className="suggestions-label">💡</span>
                {wordSuggestions.map(word => (
                  <button
                    key={word}
                    className="suggestion-chip"
                    onClick={() => acceptSuggestion(word)}
                  >
                    {word}
                  </button>
                ))}
              </div>
            )}

            <div className="sentence-controls">
              <button
                className="btn btn-small btn-primary"
                onClick={addLetterToSentence}
                disabled={(() => {
                  const mThresh = getManualAddThreshold();
                  if (prediction.mode === 'gestures') {
                    return !prediction.gesture || prediction.gesture_confidence <= mThresh;
                  }
                  return !prediction.letter || prediction.confidence <= mThresh;
                })()}
              >
                + Add {prediction.mode === 'gestures' ? 'Word' : 'Letter'}
              </button>
              <button
                className="btn btn-small"
                onClick={() => setSentence(prev => prev + ' ')}
              >
                Space
              </button>
              <button
                className="btn btn-small"
                onClick={() => setSentence('')}
                disabled={!sentence}
              >
                Clear
              </button>
              <button
                className="btn btn-small"
                onClick={copySentence}
                disabled={!sentence}
              >
                📋 Copy
              </button>
              <button
                className={`btn btn-small btn-speak ${isSpeaking ? 'speaking' : ''}`}
                onClick={isSpeaking ? stopSpeaking : speakSentence}
                disabled={!sentence}
              >
                {isSpeaking ? '⏹️ Stop' : '🔊 Speak'}
              </button>
              <button
                className="btn btn-small btn-save-session"
                onClick={() => {
                  if (saveSession()) {
                    alert('Session saved!')
                  }
                }}
                disabled={!sentence && predictionHistory.length === 0}
              >
                💾 Save
              </button>
            </div>

            <div className="shortcut-hints">
              <span><kbd>Enter</kbd> Add</span>
              <span><kbd>Space</kbd> Space</span>
              <span><kbd>⌫</kbd> Delete</span>
              <span><kbd>X</kbd> Clear</span>
            </div>
          </div>

          {/* Landmark Graph */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Fingertip Movement</span>
              <span className="card-badge">LIVE</span>
            </div>

            <div className="graph-container">
              <canvas
                ref={canvasRef}
                className="graph-canvas"
                width={340}
                height={180}
              />
            </div>

            <div className="graph-legend">
              <div className="legend-item">
                <span className="legend-dot" style={{ background: '#ef4444' }}></span>
                Thumb
              </div>
              <div className="legend-item">
                <span className="legend-dot" style={{ background: '#f59e0b' }}></span>
                Index
              </div>
              <div className="legend-item">
                <span className="legend-dot" style={{ background: '#10b981' }}></span>
                Middle
              </div>
              <div className="legend-item">
                <span className="legend-dot" style={{ background: '#6366f1' }}></span>
                Ring
              </div>
              <div className="legend-item">
                <span className="legend-dot" style={{ background: '#8b5cf6' }}></span>
                Pinky
              </div>
            </div>
          </div>

          {/* Confusion Matrix / Prediction History */}
          <div className="card confusion-matrix-card">
            <div className="card-header">
              <span className="card-title">📊 Prediction History</span>
              <span className="card-badge">{predictionHistory.length} samples</span>
            </div>

            {predictionHistory.length > 0 ? (
              <>
                {(() => {
                  const hist = predictionHistory
                  const freq = hist.reduce((acc, p) => {
                    acc[p.letter] = (acc[p.letter] || 0) + 1
                    return acc
                  }, {})
                  const confusion = {}
                  hist.forEach(p => {
                    const trueLabel = p.letter
                      ; (p.top3 || []).forEach(alt => {
                        if (alt.letter && alt.letter !== trueLabel) {
                          confusion[trueLabel] = confusion[trueLabel] || {}
                          confusion[trueLabel][alt.letter] = (confusion[trueLabel][alt.letter] || 0) + 1
                        }
                      })
                  })
                  const confusedPairs = Object.entries(confusion).flatMap(([t, alts]) =>
                    Object.entries(alts).map(([a, c]) => [t, a, c])
                  ).sort((a, b) => b[2] - a[2]).slice(0, 8)
                  const maxFreq = Math.max(...Object.values(freq), 1)
                  const maxConf = Math.max(...confusedPairs.map(x => x[2]), 1)
                  return (
                    <>
                      <div className="letter-frequency-grid">
                        {Object.entries(
                          predictionHistory.reduce((acc, p) => {
                            acc[p.letter] = (acc[p.letter] || 0) + 1
                            return acc
                          }, {})
                        )
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 12)
                          .map(([letter, count]) => (
                            <div key={letter} className="frequency-item">
                              <div className="frequency-letter">{letter}</div>
                              <div className="frequency-bar">
                                <div
                                  className="frequency-fill"
                                  style={{ width: `${(count / maxFreq) * 100}%` }}
                                ></div>
                              </div>
                              <div className="frequency-count">{count}</div>
                            </div>
                          ))
                        }
                      </div>
                      {confusedPairs.length > 0 && (
                        <div className="confusion-pairs">
                          <div className="confusion-label">Top confused pairs:</div>
                          <div className="confusion-grid">
                            {confusedPairs.map(([t, a, c], i) => (
                              <div key={i} className="confusion-pair">
                                <span className="confusion-true">{t}</span>
                                <span className="confusion-arrow">→</span>
                                <span className="confusion-alt">{a}</span>
                                <span className="confusion-count">×{c}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      <div className="avg-confidence">
                        Avg Confidence: {(predictionHistory.reduce((sum, p) => sum + p.confidence, 0) / predictionHistory.length * 100).toFixed(0)}%
                      </div>
                    </>
                  )
                })()}
              </>
            ) : (
              <div className="no-data">Add letters to see prediction stats</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}


import InteractiveBackground from './InteractiveBackground'

function App() {
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark'
  })
  const { isOnline } = useBackendStatus()

  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  return (
    <div className="app">
      {!isOnline && (
        <div className="backend-offline-banner">
          <span className="offline-icon">⚠️</span>
          Server disconnected. Make sure the backend is running on port 5001. Reconnecting...
        </div>
      )}
      <InteractiveBackground />
      <Routes>
        <Route path="/" element={<HomePage onSettingsOpen={() => setSettingsOpen(true)} />} />
        <Route
          path="/tracker"
          element={
            <TrackerPage
              theme={theme}
              onSettingsOpen={() => setSettingsOpen(true)}
            />
          }
        />

        <Route path="/transcriber" element={<Transcriber onSettingsOpen={() => setSettingsOpen(true)} />} />
        <Route path="/enhanced" element={<EnhancedPage theme={theme} />} />
        <Route path="/guide" element={<GuidePage onSettingsOpen={() => setSettingsOpen(true)} />} />
        <Route path="/history" element={<HistoryPage onSettingsOpen={() => setSettingsOpen(true)} />} />
        <Route path="/quiz" element={<QuizPage onSettingsOpen={() => setSettingsOpen(true)} />} />
        <Route path="/practice" element={<PracticePage onSettingsOpen={() => setSettingsOpen(true)} />} />
        <Route path="/privacy" element={<PrivacyPolicy />} />
        <Route path="/terms" element={<TermsOfService />} />
        <Route path="/cookies" element={<CookiePolicy />} />
      </Routes>

      <Settings
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        theme={theme}
        onThemeChange={setTheme}
      />
    </div>
  )
}

export default App
