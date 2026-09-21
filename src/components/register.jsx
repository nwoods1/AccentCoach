import React, { useState } from 'react'
import { Navigate, Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/authContext'
import { doCreateUserWithEmailAndPassword } from '../firebase/auth'
import Parrot from '../img/mascot-parrot.png'

const Register = () => {

    const navigate = useNavigate()

    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [confirmPassword, setconfirmPassword] = useState('')
    const [isRegistering, setIsRegistering] = useState(false)
    const [errorMessage, setErrorMessage] = useState('')

    const { userLoggedIn } = useAuth()

    const onSubmit = async (e) => {
        e.preventDefault()
        if(!isRegistering) {
            setIsRegistering(true)
            await doCreateUserWithEmailAndPassword(email, password)
        }
    }

    return (
        <>
            {userLoggedIn && (<Navigate to={'/home'} replace={true} />)}

            <main className="min-h-screen w-full flex items-center justify-center px-4 py-10 bg-gradient-to-b from-slate-100 to-slate-200 dark:from-slate-950 dark:to-slate-900">
                <div className="w-full max-w-md space-y-6 card">
                    <div className="flex flex-col items-center text-center">
                        <img
                            src={Parrot}
                            alt="Accent Coach logo"
                            className="h-16 w-16 rounded-full object-cover object-top ring-2 ring-brand-400/70 shadow-sm mb-3"
                            loading="eager"
                        />
                        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
                            Create a new account
                        </h1>
                        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
                            Start practicing in a couple of minutes
                        </p>
                    </div>

                    <form
                        onSubmit={onSubmit}
                        className="space-y-4"
                    >
                        <div>
                            <label className="field-label">
                                Email
                            </label>
                            <input
                                type="email"
                                autoComplete='email'
                                required
                                value={email} onChange={(e) => { setEmail(e.target.value) }}
                                className="input"
                            />
                        </div>

                        <div>
                            <label className="field-label">
                                Password
                            </label>
                            <input
                                disabled={isRegistering}
                                type="password"
                                autoComplete='new-password'
                                required
                                value={password} onChange={(e) => { setPassword(e.target.value) }}
                                className="input"
                            />
                        </div>

                        <div>
                            <label className="field-label">
                                Confirm Password
                            </label>
                            <input
                                disabled={isRegistering}
                                type="password"
                                autoComplete='off'
                                required
                                value={confirmPassword} onChange={(e) => { setconfirmPassword(e.target.value) }}
                                className="input"
                            />
                        </div>

                        {errorMessage && (
                            <div className="alert-danger">{errorMessage}</div>
                        )}

                        <button
                            type="submit"
                            disabled={isRegistering}
                            className="btn-primary w-full"
                        >
                            {isRegistering ? 'Signing Up…' : 'Sign Up'}
                        </button>
                        <p className="text-center text-sm text-slate-600 dark:text-slate-400">
                            Already have an account?{' '}
                            <Link to={'/login'} className="font-semibold text-brand-600 dark:text-brand-400 hover:underline">
                                Continue
                            </Link>
                        </p>
                    </form>
                </div>
            </main>
        </>
    )
}

export default Register
