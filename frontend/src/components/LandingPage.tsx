/**
 * Landing Page - Amfonica static landing page
 */
import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css';

function LandingPage() {
  const navRef = useRef<HTMLElement>(null);

  useEffect(() => {
    // Smooth scroll for navigation
    const handleAnchorClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const anchor = target.closest('a[href^="#"]');
      if (anchor) {
        e.preventDefault();
        const href = anchor.getAttribute('href');
        if (href) {
          const targetElement = document.querySelector(href);
          if (targetElement) {
            targetElement.scrollIntoView({
              behavior: 'smooth',
              block: 'start'
            });
          }
        }
      }
    };

    document.addEventListener('click', handleAnchorClick);

    // Add scroll-based nav background
    const handleScroll = () => {
      if (navRef.current) {
        if (window.scrollY > 50) {
          navRef.current.style.background = 'rgba(250, 250, 249, 0.95)';
        } else {
          navRef.current.style.background = 'rgba(250, 250, 249, 0.85)';
        }
      }
    };

    window.addEventListener('scroll', handleScroll);

    // Intersection Observer for fade-in animations
    const observerOptions = {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          (entry.target as HTMLElement).style.opacity = '1';
          (entry.target as HTMLElement).style.transform = 'translateY(0)';
        }
      });
    }, observerOptions);

    document.querySelectorAll('.feature-card, .use-case-category, .privacy-card, .mode-card').forEach(el => {
      (el as HTMLElement).style.opacity = '0';
      (el as HTMLElement).style.transform = 'translateY(20px)';
      (el as HTMLElement).style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
      observer.observe(el);
    });

    return () => {
      document.removeEventListener('click', handleAnchorClick);
      window.removeEventListener('scroll', handleScroll);
      observer.disconnect();
    };
  }, []);

  return (
    <div className="landing-page">
      {/* Navigation */}
      <nav ref={navRef}>
        <div className="container">
          <a href="#" className="logo">
            <div className="logo-mark"></div>
            <div className="logo-text">Amfonica</div>
          </a>
          <ul>
            <li><a href="#features">Features</a></li>
            <li><a href="#use-cases">Use Cases</a></li>
            <li><a href="#privacy">Privacy</a></li>
            <li><a href="#modes">Modes</a></li>
            <li><Link to="/login" className="nav-cta">Get Started</Link></li>
          </ul>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="container">
          <div className="hero-grid">
            <div className="hero-content">
              <div className="hero-badge">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
                Powered by Expert Frameworks
              </div>
              <h1>Your AI Edge <em>at the Table</em></h1>
              <p className="hero-description">
                An AI negotiation partner that applies proven strategies from world-leading experts to your real deals. Prepare smarter. Respond sharper. Close confidently.
              </p>
              <div className="hero-actions">
                <Link to="/login" className="btn btn-primary">
                  Start Negotiating
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M5 12h14M12 5l7 7-7 7"/>
                  </svg>
                </Link>
                <a href="#features" className="btn btn-secondary">See How It Works</a>
              </div>
            </div>
            <div className="hero-visual">
              <div className="hero-card">
                <div className="chat-message">
                  <div className="chat-avatar user">You</div>
                  <div className="chat-content">
                    <div className="chat-label">Your Scenario</div>
                    <div className="chat-text">They just said "that's our final offer" on the salary. What do I do?</div>
                  </div>
                </div>
                <div className="chat-message">
                  <div className="chat-avatar ai">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="3"/>
                      <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                    </svg>
                  </div>
                  <div className="chat-content">
                    <div className="chat-label">Amfonica</div>
                    <div className="chat-text">
                      <strong>Don't accept "final" at face value.</strong> Try a calibrated question: "How am I supposed to accept that when the market rate is 15% higher?" This shifts the burden without confrontation. Also explore...
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Trusted By / Frameworks */}
      <section className="trusted-by">
        <div className="container">
          <p>Built on Proven Negotiation Science</p>
          <div className="frameworks">
            <div className="framework">
              <div className="framework-icon">📘</div>
              <div className="framework-name">Getting to Yes</div>
              <div className="framework-author">Fisher & Ury</div>
            </div>
            <div className="framework">
              <div className="framework-icon">🎯</div>
              <div className="framework-name">Never Split the Difference</div>
              <div className="framework-author">Chris Voss</div>
            </div>
            <div className="framework">
              <div className="framework-icon">🧠</div>
              <div className="framework-name">Influence</div>
              <div className="framework-author">Robert Cialdini</div>
            </div>
            <div className="framework">
              <div className="framework-icon">⚖️</div>
              <div className="framework-name">Getting Past No</div>
              <div className="framework-author">William Ury</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="features" id="features">
        <div className="container">
          <div className="section-header">
            <div className="section-label">Features</div>
            <h2>Everything You Need to Negotiate Like a Pro</h2>
            <p>From preparation to real-time coaching, Amfonica has you covered at every stage.</p>
          </div>
          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-icon">👤</div>
              <h3>Multiple Personas</h3>
              <p>Create profiles for yourself and your counterparts. Switch contexts instantly between different negotiations.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📁</div>
              <h3>Negotiation Management</h3>
              <p>Track multiple active negotiations. Personal deals, professional contracts, complex multi-party discussions—all organized.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">📄</div>
              <h3>Document Upload</h3>
              <p>Upload contracts, emails, and background docs. Amfonica analyzes context to give you targeted advice.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🎓</div>
              <h3>Expert Frameworks</h3>
              <p>Powered by RAG with content from the world's top negotiation books. Real expertise, not generic AI responses.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Real-Time & Async</h3>
              <p>Get instant responses during live negotiations, or use async mode to plan and analyze your strategy in depth.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔒</div>
              <h3>Privacy First</h3>
              <p>Use your own API keys, powerful free models, or run 100% locally. Your negotiations stay your business.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="use-cases" id="use-cases">
        <div className="container">
          <div className="section-header">
            <div className="section-label">Use Cases</div>
            <h2>For Every Negotiation in Your Life</h2>
            <p>Whether it's personal or professional, high-stakes or everyday, Amfonica adapts to your situation.</p>
          </div>
          <div className="use-case-grid">
            <div className="use-case-category">
              <div className="use-case-header">
                <div className="use-case-icon personal">🏠</div>
                <div>
                  <h3>Personal</h3>
                  <span>Life's big moments deserve preparation</span>
                </div>
              </div>
              <div className="use-case-list">
                <div className="use-case-item">
                  <div className="use-case-check">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <p><strong>Buying a car</strong> — Get the best price without the dealer games</p>
                </div>
                <div className="use-case-item">
                  <div className="use-case-check">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <p><strong>Rent negotiations</strong> — Lower your rent or get better lease terms</p>
                </div>
                <div className="use-case-item">
                  <div className="use-case-check">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <p><strong>Home purchase</strong> — Navigate offers, counteroffers, and contingencies</p>
                </div>
                <div className="use-case-item">
                  <div className="use-case-check">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <p><strong>Service disputes</strong> — Handle contractors, bills, and warranties</p>
                </div>
              </div>
            </div>
            <div className="use-case-category">
              <div className="use-case-header">
                <div className="use-case-icon professional">💼</div>
                <div>
                  <h3>Professional</h3>
                  <span>Close deals with confidence</span>
                </div>
              </div>
              <div className="use-case-list">
                <div className="use-case-item">
                  <div className="use-case-check">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <p><strong>Salary & job offers</strong> — Maximize compensation packages</p>
                </div>
                <div className="use-case-item">
                  <div className="use-case-check">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <p><strong>Sales deals</strong> — Close faster with better terms</p>
                </div>
                <div className="use-case-item">
                  <div className="use-case-check">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <p><strong>Vendor contracts</strong> — Negotiate SLAs, pricing, and terms</p>
                </div>
                <div className="use-case-item">
                  <div className="use-case-check">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <p><strong>Partnership agreements</strong> — Structure win-win deals</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Privacy */}
      <section className="privacy" id="privacy">
        <div className="container">
          <div className="section-header">
            <div className="section-label">Your Data, Your Rules</div>
            <h2>Complete Control Over Your AI</h2>
            <p>Choose exactly how Amfonica runs. From cloud-powered to fully local—you decide.</p>
          </div>
          <div className="privacy-grid">
            <div className="privacy-card">
              <div className="privacy-icon">🔑</div>
              <h3>Bring Your Own Keys</h3>
              <p>Use your OpenAI, Anthropic, or other API keys for premium models</p>
            </div>
            <div className="privacy-card">
              <div className="privacy-icon">🆓</div>
              <h3>Powerful Free Models</h3>
              <p>Access Ollama, DeepSeek, and other capable open models at no cost</p>
            </div>
            <div className="privacy-card">
              <div className="privacy-icon">💻</div>
              <h3>100% Local</h3>
              <p>Run entirely on your machine. Nothing leaves your computer. Ever.</p>
            </div>
            <div className="privacy-card">
              <div className="privacy-icon">📚</div>
              <h3>Your Documents</h3>
              <p>Upload your own negotiation resources to build a custom knowledge base</p>
            </div>
          </div>
        </div>
      </section>

      {/* Modes */}
      <section className="modes" id="modes">
        <div className="container">
          <div className="section-header">
            <div className="section-label">Two Powerful Modes</div>
            <h2>Real-Time Coaching or Strategic Planning</h2>
            <p>Use Amfonica in the moment or step back to plan your approach.</p>
          </div>
          <div className="modes-grid">
            <div className="mode-card sync">
              <div className="mode-badge">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <polyline points="12 6 12 12 16 14"/>
                </svg>
                Synchronous Mode
              </div>
              <h3>Real-Time Support</h3>
              <p>Get instant advice during live negotiations. Perfect for calls, meetings, and rapid exchanges.</p>
              <div className="mode-features">
                <div className="mode-feature">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  Instant response suggestions
                </div>
                <div className="mode-feature">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  Counter-argument preparation
                </div>
                <div className="mode-feature">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  Quick tactical pivots
                </div>
              </div>
            </div>
            <div className="mode-card async">
              <div className="mode-badge">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                  <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                </svg>
                Asynchronous Mode
              </div>
              <h3>Strategic Planning</h3>
              <p>Take time to analyze positions, build strategies, and prepare comprehensive game plans.</p>
              <div className="mode-features">
                <div className="mode-feature">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  Deep scenario analysis
                </div>
                <div className="mode-feature">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  BATNA development
                </div>
                <div className="mode-feature">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  Strategy documentation
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta">
        <div className="container">
          <div className="cta-card">
            <h2>Ready to Win More Deals?</h2>
            <p>Start negotiating smarter today. No credit card required for free model access.</p>
            <div className="cta-actions">
              <Link to="/login" className="btn btn-gold">
                Get Started Free
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7"/>
                </svg>
              </Link>
              <a href="#" className="btn btn-outline-white">View Documentation</a>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer>
        <div className="container">
          <div className="footer-grid">
            <div className="footer-brand">
              <a href="#" className="logo">
                <div className="logo-mark"></div>
                <div className="logo-text">Amfonica</div>
              </a>
              <p>Your AI edge at the table. Powered by expert frameworks and designed for real-world results.</p>
            </div>
            <div className="footer-col">
              <h4>Product</h4>
              <ul>
                <li><a href="#">Features</a></li>
                <li><a href="#">Use Cases</a></li>
                <li><a href="#">Pricing</a></li>
                <li><a href="#">Roadmap</a></li>
              </ul>
            </div>
            <div className="footer-col">
              <h4>Resources</h4>
              <ul>
                <li><a href="#">Documentation</a></li>
                <li><a href="#">API Reference</a></li>
                <li><a href="#">Self-Hosting Guide</a></li>
                <li><a href="#">Blog</a></li>
              </ul>
            </div>
            <div className="footer-col">
              <h4>Company</h4>
              <ul>
                <li><a href="#">About</a></li>
                <li><a href="#">Contact</a></li>
                <li><a href="#">Privacy Policy</a></li>
                <li><a href="#">Terms of Service</a></li>
              </ul>
            </div>
          </div>
          <div className="footer-bottom">
            <p>© 2025 Amfonica. All rights reserved.</p>
            <div className="footer-social">
              <a href="#" aria-label="GitHub">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </a>
              <a href="#" aria-label="Twitter">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                </svg>
              </a>
              <a href="#" aria-label="LinkedIn">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default LandingPage;
