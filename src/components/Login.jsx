import React, { useState } from 'react'
import { Navigate, Link, useNavigate } from 'react-router-dom'
import { doSignInWithEmailAndPassword, doSignInWithGoogle } from '../firebase/auth'
import { auth } from '../firebase/firebase'
import { signInAnonymously } from 'firebase/auth'
import { useAuth } from '../contexts/authContext'
import Parrot from '../img/mascot-parrot.png'

const Login = () => {
  const { userLoggedIn } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isSigningIn, setIsSigningIn] = useState(false);

  const onSubmit = async (e) => {
    e.preventDefault();
    if (isSigningIn) return;
    setErrorMessage('');
    setIsSigningIn(true);
    try {
      await doSignInWithEmailAndPassword(email, password);
      navigate('/', { replace: true });
    } catch (err) {
      setErrorMessage(err?.message || 'Sign in failed');
    } finally {
      setIsSigningIn(false);
    }
  };

  const onGoogleSignIn = async (e) => {
    e.preventDefault();
    if (isSigningIn) return;
    setErrorMessage('');
    setIsSigningIn(true);
    try {
      await doSignInWithGoogle();
      navigate('/', { replace: true });
    } catch (err) {
      setErrorMessage(err?.message || 'Google sign-in failed');
    } finally {
      setIsSigningIn(false);
    }
  };

  const onGuestSignIn = async (e) => {
    e.preventDefault();
    if (isSigningIn) return;
    setErrorMessage('');
    setIsSigningIn(true);
    try {
      await signInAnonymously(auth);
      navigate('/', { replace: true });
    } catch (err) {
      setErrorMessage(err?.message || 'Guest sign-in failed');
    } finally {
      setIsSigningIn(false);
    }
  };

  return (
    <div>
      {userLoggedIn && (<Navigate to="/home" replace />)}

      <main className="min-h-screen w-full flex items-center justify-center px-4 py-10 bg-gradient-to-b from-slate-100 to-slate-200 dark:from-slate-950 dark:to-slate-900">
        <div className="w-full max-w-md space-y-6 card">
          {/* Brand header */}
          <div className="flex flex-col items-center text-center">
            <img
              src={Parrot}
              alt="Accent Coach logo"
              className="h-16 w-16 rounded-full object-cover object-top ring-2 ring-brand-400/70 shadow-sm mb-3"
              loading="eager"
            />
            <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
              Accent Coach
            </h1>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Welcome back</p>
          </div>

          <form onSubmit={onSubmit} className="space-y-4">
            <div>
              <label className="field-label">Email</label>
              <input
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => { setEmail(e.target.value); setErrorMessage(''); }}
                className="input"
              />
            </div>

            <div>
              <label className="field-label">Password</label>
              <input
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => { setPassword(e.target.value); setErrorMessage(''); }}
                className="input"
              />
            </div>

            {errorMessage && <div className="alert-danger">{errorMessage}</div>}

            <button type="submit" disabled={isSigningIn} className="btn-primary w-full">
              {isSigningIn ? 'Signing In…' : 'Sign In'}
            </button>
          </form>

          <p className="text-center text-sm text-slate-600 dark:text-slate-400">
            Don&apos;t have an account?{' '}
            <Link to="/register" className="font-semibold text-brand-600 dark:text-brand-400 hover:underline">
              Sign up
            </Link>
          </p>

          <div className="divider-label">OR</div>

          <div className="space-y-2">
            <button
              disabled={isSigningIn}
              onClick={onGoogleSignIn}
              className="btn-outline w-full"
            >
              {isSigningIn ? 'Signing In…' : 'Continue with Google'}
            </button>

            <button
              disabled={isSigningIn}
              onClick={onGuestSignIn}
              className="btn-ghost w-full border border-slate-300 dark:border-slate-700"
              title="Use the app without saving progress"
            >
              {isSigningIn ? 'Signing In…' : 'Continue as guest'}
            </button>
          </div>

          <p className="text-center text-xs text-slate-500 dark:text-slate-400">
            Guest mode won&rsquo;t save your progress.
          </p>
        </div>
      </main>
    </div>
  );
};

export default Login;
