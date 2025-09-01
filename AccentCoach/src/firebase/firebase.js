import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyB66gAs4-HcPhxJDzUO8xQkZSEWg7zb0OA",
  authDomain: "accentcoach-7dacc.firebaseapp.com",
  projectId: "accentcoach-7dacc",
  storageBucket: "accentcoach-7dacc.firebasestorage.app",
  messagingSenderId: "892495373061",
  appId: "1:892495373061:web:ace722307eb5165bc2cd6c",
  measurementId: "G-JLXEPC98MK"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
const auth = getAuth(app);
const db = getFirestore(app);

export { app, auth, db };
export default app;