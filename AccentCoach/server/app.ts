import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import uploadRoute from './routes/upload';

dotenv.config();

const app = express();
app.use(cors());
app.use('/upload', uploadRoute);

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
