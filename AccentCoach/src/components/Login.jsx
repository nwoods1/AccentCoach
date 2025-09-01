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

      <main className="w-full h-screen flex items-center justify-center px-4">
        <div className="w-full max-w-md text-gray-600 space-y-5 p-6 shadow-xl border rounded-xl bg-white dark:bg-slate-900 dark:text-slate-200 dark:border-slate-700">
          {/* Brand header */}
          <div className="flex flex-col items-center text-center">
            <img
              src={Parrot}
              alt="Accent Coach logo"
              className="h-24 w-16 mb-2 drop-shadow"
              loading="eager"
            />
            <h1 className="text-2xl font-extrabold tracking-tight text-gray-900 dark:text-slate-100">
              Accent Coach
            </h1>
            <p className="mt-1 text-sm text-gray-500 dark:text-slate-400">Welcome back</p>
          </div>

          <form onSubmit={onSubmit} className="space-y-5">
            <div>
              <label className="text-sm text-gray-600 dark:text-slate-300 font-bold">Email</label>
              <input
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => { setEmail(e.target.value); setErrorMessage(''); }}
                className="w-full mt-2 px-3 py-2 text-gray-700 dark:text-slate-100 bg-transparent outline-none border focus:border-yellow-600 shadow-sm rounded-lg transition duration-300 dark:border-slate-700"
              />
            </div>

            <div>
              <label className="text-sm text-gray-600 dark:text-slate-300 font-bold">Password</label>
              <input
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => { setPassword(e.target.value); setErrorMessage(''); }}
                className="w-full mt-2 px-3 py-2 text-gray-700 dark:text-slate-100 bg-transparent outline-none border focus:border-yellow-600 shadow-sm rounded-lg transition duration-300 dark:border-slate-700"
              />
            </div>

            {errorMessage && <span className="block text-red-600 font-bold">{errorMessage}</span>}

            <button
              type="submit"
              disabled={isSigningIn}
              className={`w-full px-4 py-2 text-white font-medium rounded-lg ${
                isSigningIn ? 'bg-gray-300 cursor-not-allowed' : 'bg-yellow-600 hover:bg-yellow-700 hover:shadow-xl transition duration-300'
              }`}
            >
              {isSigningIn ? 'Signing In...' : 'Sign In'}
            </button>
          </form>

          <p className="text-center text-sm">
            Don&apos;t have an account?{' '}
            <Link to="/register" className="hover:underline font-bold">Sign up</Link>
          </p>

          <div className="flex flex-row items-center w-full">
            <div className="border-b mb-2.5 mr-2 w-full dark:border-slate-700" />
            <div className="text-sm font-bold w-fit">OR</div>
            <div className="border-b mb-2.5 ml-2 w-full dark:border-slate-700" />
          </div>

          <button
            disabled={isSigningIn}
            onClick={onGoogleSignIn}
            className={`w-full flex items-center justify-center gap-x-3 py-2.5 border rounded-lg text-sm font-medium ${
              isSigningIn ? 'cursor-not-allowed' : 'hover:bg-gray-100 transition duration-300 active:bg-gray-100 dark:hover:bg-slate-800'
            } dark:border-slate-700`}
          >
            {isSigningIn ? 'Signing In...' : 'Continue with Google'}
          </button>

          <button
            disabled={isSigningIn}
            onClick={onGuestSignIn}
            className={`w-full mt-2 flex items-center justify-center gap-x-3 py-2.5 border rounded-lg text-sm font-medium ${
              isSigningIn ? 'cursor-not-allowed' : 'hover:bg-gray-100 transition duration-300 active:bg-gray-100 dark:hover:bg-slate-800'
            } dark:border-slate-700`}
            title="Use the app without saving progress"
          >
            {isSigningIn ? 'Signing In...' : 'Continue as guest'}
          </button>

          <p className="text-center text-xs text-gray-500 dark:text-slate-400">
            Guest mode won’t save your progress.
          </p>
        </div>
      </main>
    </div>
  );
};

export default Login;
