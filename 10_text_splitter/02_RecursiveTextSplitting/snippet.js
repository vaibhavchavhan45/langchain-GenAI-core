const express = require('express');
const router = express.Router();

router.get('/api/users', authMiddleware, (req, res) => {
    const users = [
        { id: 1, name: 'Rahul', email: 'rahul@example.com' },
        { id: 2, name: 'Priya', email: 'priya@example.com' },
        { id: 3, name: 'Amit', email: 'amit@example.com' }
    ];
    res.status(200).json({ success: true, data: users });
});

router.post('/api/users', (req, res) => {
    const { name, email, password } = req.body;
    
    if (!name || !email || !password) {
        return res.status(400).json({ error: 'Missing required fields' });
    }
    
    const newUser = {
        id: Math.floor(Math.random() * 1000),
        name,
        email,
        createdAt: new Date().toISOString()
    };
    
    res.status(201).json({ success: true, data: newUser });
});

module.exports = router;