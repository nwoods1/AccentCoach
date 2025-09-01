import express, { Request, Response } from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { exec } from 'child_process';

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.post('/', upload.single('audio'), async (req: Request, res: Response): Promise<void> => {
  if (!req.file) {
    res.status(400).json({ error: 'No file uploaded.' });
    return;
  }

  const originalPath = path.join(__dirname, '..', req.file.path);
  const renamedPath = originalPath + '.webm';
  const scriptPath = path.join(__dirname, '..', '..', 'analyzer', 'analyze.py');
  const expected = "Hello, my name is John.";

  try {
    fs.renameSync(originalPath, renamedPath);

    const command = `python "${scriptPath}" "${renamedPath}" "${expected}"`;
    exec(command, (error, stdout) => {
      fs.unlinkSync(renamedPath);

      if (error) {
        console.error("Python script error:", error);
        res.status(500).json({ error: 'Python script failed.' });
        return;
      }

      try {
        const result = JSON.parse(stdout);
        res.json(result);
      } catch (err) {
        console.error("Output parse error:", err);
        res.status(500).json({ error: 'Failed to parse Python output.' });
      }
    });
  } catch (err) {
    console.error("File handling error:", err);
    res.status(500).json({ error: 'Internal server error.' });
  }
});

export default router;
