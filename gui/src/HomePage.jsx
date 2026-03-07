import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import './App.css'

const TITLE_TEXT = 'SignMate'
const TYPING_SPEED = 60

const FAQ_ITEMS = [
    {
        question: 'What is SignMate and how does it work?',
        answer: 'SignMate uses your webcam and AI-powered computer vision to detect hand signs in real-time. It tracks 21 landmarks per hand using MediaPipe technology and translates American Sign Language (ASL) gestures into text instantly.'
    },
    {
        question: 'Do I need to install anything?',
        answer: 'SignMate runs in your browser — no plugins or extensions needed. You just need a webcam and a modern browser (Chrome, Firefox, or Edge). The backend server handles all the AI processing locally on your machine.'
    },
    {
        question: 'Is my camera data private?',
        answer: 'Absolutely. All video processing happens locally on your device. Your camera feed is never uploaded to any server or stored anywhere. We take privacy seriously — see our Privacy Policy for full details.'
    },
    {
        question: 'How accurate is the translation?',
        answer: 'SignMate achieves high accuracy for static ASL fingerspelling (A-Z letters and 0-9 numbers). The AI model is constantly being improved. You can see real-time confidence scores for each prediction as you sign.'
    },
    {
        question: 'Can I use SignMate to learn sign language?',
        answer: 'Yes! SignMate includes a Practice mode where you can learn individual signs, a Quiz mode to test your knowledge, and a comprehensive Guide with visual references for each letter and number.'
    },
    {
        question: 'Is SignMate free to use?',
        answer: 'SignMate is completely free and open-source. You can use all features — real-time translation, video transcription, practice tools, and quizzes — without any cost or account required.'
    }
]

function HomePage({ onSettingsOpen }) {
    const navigate = useNavigate()
    const [openFaq, setOpenFaq] = useState(null)
    const stepsRef = useRef(null)
    const lineRef = useRef(null)

    const toggleFaq = (index) => {
        setOpenFaq(openFaq === index ? null : index)
    }

    // Scroll-driven line animation
    useEffect(() => {
        const handleScroll = () => {
            if (!stepsRef.current || !lineRef.current) return
            const container = stepsRef.current
            const rect = container.getBoundingClientRect()
            const windowH = window.innerHeight
            // Start filling when the top of the steps enters the viewport center
            // Finish when the bottom of the steps reaches the viewport center
            const start = rect.top - windowH * 0.5
            const end = rect.bottom - windowH * 0.5
            const raw = Math.min(1, Math.max(0, -start / (end - start)))
            // Gentle ease: blend linear with smoothstep for subtle accel/decel
            const smoothstep = raw * raw * (3 - 2 * raw)
            const eased = raw * 0.4 + smoothstep * 0.6
            lineRef.current.style.setProperty('--hiw-progress', eased)
        }
        window.addEventListener('scroll', handleScroll, { passive: true })
        handleScroll() // initial
        return () => window.removeEventListener('scroll', handleScroll)
    }, [])

    const [displayedTitle, setDisplayedTitle] = useState(() =>
        typeof sessionStorage !== 'undefined' && sessionStorage.getItem('slt_typing_seen') ? TITLE_TEXT : ''
    )
    const [showCursor, setShowCursor] = useState(() =>
        typeof sessionStorage !== 'undefined' ? !sessionStorage.getItem('slt_typing_seen') : true
    )

    useEffect(() => {
        const seen = sessionStorage.getItem('slt_typing_seen')
        if (seen) return

        let i = 0
        const interval = setInterval(() => {
            if (i < TITLE_TEXT.length) {
                setDisplayedTitle(TITLE_TEXT.slice(0, i + 1))
                i++
            } else {
                clearInterval(interval)
                sessionStorage.setItem('slt_typing_seen', '1')
                const cursorInterval = setInterval(() => setShowCursor((c) => !c), 530)
                setTimeout(() => {
                    clearInterval(cursorInterval)
                    setShowCursor(false)
                }, 1500)
            }
        }, TYPING_SPEED)
        return () => clearInterval(interval)
    }, [])

    return (
        <div className="homepage">
            {/* Navigation */}
            <header className="home-header">
                <div className="home-header-inner">
                    <button className="home-logo" onClick={() => navigate('/')}>
                        <img src="/signmate_logo.png" alt="SignMate Logo" className="logo-icon" />
                        SignMate
                    </button>
                    <nav className="home-nav-pill">
                        <button onClick={() => navigate('/tracker')}>Translator</button>
                        <button onClick={() => navigate('/guide')}>Guide</button>
                        <button onClick={() => navigate('/practice')}>Practice</button>
                    </nav>
                    <div className="home-header-right">
                        <button className="home-settings-icon" onClick={onSettingsOpen} aria-label="Settings">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="3" />
                                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                            </svg>
                        </button>
                        <button className="home-cta-nav" onClick={() => navigate('/tracker')}>
                            Get Started
                        </button>
                    </div>
                </div>
            </header>

            {/* Hero Section */}
            <section className="hero hero-centered">
                <div className="hero-content">
                    <h1 className="hero-title hero-title-oneline">
                        <span className="hero-title-text">{displayedTitle}</span>
                        {showCursor && <span className="hero-title-cursor">|</span>}
                    </h1>
                    <p className="hero-description">
                        Real-time sign language translation powered by AI and computer vision.
                        Break communication barriers instantly.
                    </p>

                    <div className="hero-cta-row">
                        <button className="hero-cta-primary" onClick={() => navigate('/tracker')}>
                            Start Translating
                            <span className="cta-arrow">→</span>
                        </button>
                        <button className="hero-cta-secondary" onClick={() => {
                            document.querySelector('.hiw-section')?.scrollIntoView({ behavior: 'smooth' })
                        }}>
                            See How It Works
                        </button>
                    </div>
                </div>
            </section>

            {/* Product Mockup */}
            <section className="product-showcase">
                <div className="product-mockup">
                    <div className="mockup-window">
                        <div className="mockup-toolbar">
                            <span className="mockup-dot red"></span>
                            <span className="mockup-dot yellow"></span>
                            <span className="mockup-dot green"></span>
                            <span className="mockup-title">SignMate — Real-Time Translator</span>
                        </div>
                        <div className="mockup-content">
                            <div className="mockup-layout">
                                <div className="mockup-video-area">
                                    <div className="mockup-video-placeholder">
                                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.4">
                                            <path d="M23 7l-7 5 7 5V7z" />
                                            <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                                        </svg>
                                        <span>Live Camera Feed</span>
                                    </div>
                                    <div className="mockup-badges">
                                        <span className="mockup-badge live">● LIVE</span>
                                        <span className="mockup-badge tracking">TRACKING</span>
                                    </div>
                                </div>
                                <div className="mockup-sidebar-area">
                                    <div className="mockup-prediction">
                                        <span className="mockup-pred-label">AI Prediction</span>
                                        <span className="mockup-pred-letter">A</span>
                                        <div className="mockup-pred-bar">
                                            <div className="mockup-pred-fill" style={{ width: '94%' }}></div>
                                        </div>
                                        <span className="mockup-pred-conf">94%</span>
                                    </div>
                                    <div className="mockup-sentence">
                                        <span className="mockup-sent-label">Sentence Builder</span>
                                        <span className="mockup-sent-text">HELLO WORLD</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* How It Works Section */}
            <section className="hiw-section">
                <div className="hiw-layout">
                    <div className="hiw-context">
                        <h2 className="hiw-title">How it works</h2>
                        <p className="hiw-subtitle">
                            From gestures to text, in the blink of an eye.
                        </p>
                        <button className="hiw-cta" onClick={() => navigate('/tracker')}>
                            Start Translating
                        </button>
                    </div>
                    <div className="hiw-steps" ref={stepsRef}>
                        <div className="hiw-line" ref={lineRef}>
                            <div className="hiw-line-fill"></div>
                        </div>
                        <div className="hiw-step">
                            <div className="hiw-step-icon">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M23 7l-7 5 7 5V7z" />
                                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                                </svg>
                            </div>
                            <div className="hiw-step-content">
                                <h3 className="hiw-step-title">Camera capture</h3>
                                <p className="hiw-step-desc">
                                    Your webcam captures a live video feed at up to 60 fps. The image is mirrored so movements feel natural, like looking in a mirror.
                                </p>
                            </div>
                        </div>

                        <div className="hiw-step">
                            <div className="hiw-step-icon">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2" />
                                    <path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2" />
                                    <path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8" />
                                    <path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 13" />
                                </svg>
                            </div>
                            <div className="hiw-step-content">
                                <h3 className="hiw-step-title">Hand detection & tracking</h3>
                                <p className="hiw-step-desc">
                                    MediaPipe identifies your hands and maps 21 precise 3D landmarks — from wrist to every fingertip — tracking position, rotation, and depth in real time.
                                </p>
                            </div>
                        </div>

                        <div className="hiw-step">
                            <div className="hiw-step-icon">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                                    <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                                    <line x1="12" y1="22.08" x2="12" y2="12" />
                                </svg>
                            </div>
                            <div className="hiw-step-content">
                                <h3 className="hiw-step-title">Feature extraction</h3>
                                <p className="hiw-step-desc">
                                    The 3D coordinates of all 42 landmarks are extracted into a compact 126-feature vector and normalized so hand size and camera distance don't affect results.
                                </p>
                            </div>
                        </div>

                        <div className="hiw-step">
                            <div className="hiw-step-icon">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z" />
                                    <line x1="10" y1="22" x2="14" y2="22" />
                                </svg>
                            </div>
                            <div className="hiw-step-content">
                                <h3 className="hiw-step-title">AI classification</h3>
                                <p className="hiw-step-desc">
                                    A deep neural network scores every possible sign (A–Z, 0–9) and returns the top 3 predictions with real-time confidence percentages.
                                </p>
                            </div>
                        </div>

                        <div className="hiw-step">
                            <div className="hiw-step-icon">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                                </svg>
                            </div>
                            <div className="hiw-step-content">
                                <h3 className="hiw-step-title">Sentence building</h3>
                                <p className="hiw-step-desc">
                                    Individual letters are stitched into words with autocomplete from a 370k-word dictionary. Pause to add spaces, and hear your message read aloud with text-to-speech.
                                </p>
                            </div>
                        </div>

                        <div className="hiw-step">
                            <div className="hiw-step-icon">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                                </svg>
                            </div>
                            <div className="hiw-step-content">
                                <h3 className="hiw-step-title">Full-word recognition</h3>
                                <p className="hiw-step-desc">
                                    Enhanced Mode uses an LSTM network that watches sequences of motion to recognize whole dynamic signs like "hello," "thank you," and "yes."
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </section >

            {/* Trust / Built With Bar */}
            < section className="trust-bar" >
                <span className="trust-label">Powered by</span>
                <div className="trust-items">
                    <span className="trust-item">Python</span>
                    <span className="trust-divider">·</span>
                    <span className="trust-item">MediaPipe</span>
                    <span className="trust-divider">·</span>
                    <span className="trust-item">React</span>
                    <span className="trust-divider">·</span>
                    <span className="trust-item">Flask</span>
                    <span className="trust-divider">·</span>
                    <span className="trust-item">OpenCV</span>
                </div>
            </section >

            {/* FAQ Section */}
            < section className="faq-section" >
                <h2 className="faq-title">Frequently Asked Questions</h2>
                <p className="faq-subtitle">Have a question? We have answers.</p>
                <div className="faq-list">
                    {FAQ_ITEMS.map((item, index) => (
                        <div
                            key={index}
                            className={`faq-item ${openFaq === index ? 'faq-item-open' : ''}`}
                        >
                            <button
                                className="faq-question"
                                onClick={() => toggleFaq(index)}
                                aria-expanded={openFaq === index}
                            >
                                <span>{item.question}</span>
                                <svg
                                    className={`faq-chevron ${openFaq === index ? 'faq-chevron-open' : ''}`}
                                    width="20"
                                    height="20"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                >
                                    <polyline points="6 9 12 15 18 9" />
                                </svg>
                            </button>
                            <div className={`faq-answer ${openFaq === index ? 'faq-answer-open' : ''}`}>
                                <p>{item.answer}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </section >

            {/* Footer */}
            < footer className="home-footer" >
                <div className="footer-content">
                    <div className="footer-brand">
                        <span className="footer-logo">SignMate</span>
                        <p>AI-Powered Sign Language Translation</p>
                        <div className="footer-social">
                            <a href="#" aria-label="GitHub">GitHub</a>
                            <a href="#" aria-label="Twitter">Twitter</a>
                            <a href="#" aria-label="Discord">Discord</a>
                        </div>
                    </div>

                    <div className="footer-links-grid">
                        <div className="footer-column">
                            <h4>Tools</h4>
                            <button onClick={() => navigate('/tracker')}>Translator</button>
                            <button onClick={() => navigate('/transcriber')}>Transcriber</button>
                            <button onClick={() => navigate('/enhanced')}>Enhanced Mode</button>
                        </div>
                        <div className="footer-column">
                            <h4>Resources</h4>
                            <button onClick={() => navigate('/guide')}>Guide</button>
                            <button onClick={() => navigate('/history')}>History</button>
                            <button onClick={() => navigate('/practice')}>Practice</button>
                            <button onClick={() => navigate('/quiz')}>Quiz</button>
                        </div>
                        <div className="footer-column">
                            <h4>Legal</h4>
                            <button onClick={() => navigate('/privacy')}>Privacy Policy</button>
                            <button onClick={() => navigate('/terms')}>Terms of Service</button>
                            <button onClick={() => navigate('/cookies')}>Cookie Policy</button>
                        </div>
                    </div>
                </div>
                <div className="footer-bottom">
                    <p>&copy; {new Date().getFullYear()} SignMate. All rights reserved.</p>
                </div>
            </footer >
        </div >
    )
}

export default HomePage
