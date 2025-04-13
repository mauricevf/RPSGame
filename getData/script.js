import { readFile, writeFile } from 'fs/promises';
import path from 'path';

// Load and parse data.json
const rawData = await readFile(path.resolve('./data.json'), 'utf-8');
const data = JSON.parse(rawData);

// Shuffle the data
data.sort(() => Math.random() - 0.5);

// Split into train and test
const train = data.slice(0, Math.floor(data.length * 0.8));
const test = data.slice(Math.floor(data.length * 0.8));

// Save the new JSON files
await writeFile('train.json', JSON.stringify(train, null, 2));
await writeFile('test.json', JSON.stringify(test, null, 2));

console.log('train.json and test.json saved successfully!');
