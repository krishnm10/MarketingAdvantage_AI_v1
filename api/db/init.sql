-- ========================================
-- MarketingAdvantage Full Initialization
-- File: init.sql
-- Combines: schema_v1.sql + seed_data.sql
-- ========================================

-- ========================================
-- MARKETINGADVANTAGE DATABASE INITIALIZER
-- Version: 1.0
-- Author: Code GPT
-- Date: 2025-10-27
-- Purpose: Create schema and populate seed data
-- ========================================

-- Drop existing tables (for clean reinitialization)
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;

SET search_path TO public;

-- ========================================
-- STEP 1: DATABASE SCHEMA DEFINITION
-- ========================================

-- USERS TABLE
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW()
);

-- BUSINESS PROFILES
CREATE TABLE business_profiles (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255),
    category VARCHAR(255),
    description TEXT,
    location VARCHAR(255),
    target_audience TEXT,
    goals JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- STRATEGIES
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    business_id INT REFERENCES business_profiles(id) ON DELETE CASCADE,
    summary TEXT,
    channels JSONB,
    uvp TEXT,
    growth_plan JSONB
);

-- COMPETITORS
CREATE TABLE competitors (
    id SERIAL PRIMARY KEY,
    business_id INT REFERENCES business_profiles(id),
    name VARCHAR(255),
    keywords JSONB,
    ad_style TEXT,
    engagement_rate FLOAT,
    notes TEXT
);

-- CONTENTS
CREATE TABLE contents (
    id SERIAL PRIMARY KEY,
    business_id INT REFERENCES business_profiles(id),
    type VARCHAR(50),
    title TEXT,
    body TEXT,
    hashtags JSONB,
    platform VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- CAMPAIGNS
CREATE TABLE campaigns (
    id SERIAL PRIMARY KEY,
    business_id INT REFERENCES business_profiles(id),
    platform VARCHAR(100),
    objective TEXT,
    headlineA TEXT,
    headlineB TEXT,
    bodyA TEXT,
    bodyB TEXT,
    cta TEXT,
    status VARCHAR(50) DEFAULT 'draft'
);

-- KPIS
CREATE TABLE kpis (
    id SERIAL PRIMARY KEY,
    business_id INT REFERENCES business_profiles(id),
    date DATE,
    impressions INT,
    clicks INT,
    conversions INT,
    ad_spend FLOAT,
    revenue FLOAT,
    roas FLOAT,
    ctr FLOAT,
    crr FLOAT
);

-- TRENDS
CREATE TABLE trends (
    id SERIAL PRIMARY KEY,
    category VARCHAR(255),
    source VARCHAR(255),
    topic TEXT,
    forecast_score FLOAT,
    date_detected DATE
);

-- ENGAGEMENT SCRIPTS
CREATE TABLE engagement_scripts (
    id SERIAL PRIMARY KEY,
    business_id INT REFERENCES business_profiles(id),
    scenario VARCHAR(255),
    message_template TEXT,
    loyalty_idea TEXT
);

-- ========================================
-- STEP 2: SEED DATA INSERTION
-- ========================================

-- USERS
INSERT INTO users (email, password_hash, role)
VALUES
('founder@ecobloom.com', 'hashed_pass_123', 'user'),
('agency@marketboost.ai', 'hashed_pass_456', 'admin');

-- BUSINESS PROFILES
INSERT INTO business_profiles (user_id, name, category, description, location, target_audience, goals)
VALUES
(1, 'EcoBloom', 'Sustainable Skincare', 
 'Eco-friendly skincare line offering plant-based serums and creams.', 
 'Los Angeles, CA', 
 'Eco-conscious women aged 20-40', 
 '["Increase brand awareness", "Boost sales through social media"]'),

(2, 'TechSpark', 'AI SaaS Startup', 
 'An AI-driven SaaS platform that automates marketing analytics for small businesses.', 
 'Austin, TX', 
 'Small business owners, marketers, and agencies', 
 '["Expand market reach", "Increase MRR", "Improve SEO ranking"]');

-- STRATEGIES
INSERT INTO strategies (business_id, summary, channels, uvp, growth_plan)
VALUES
(1,
 'Focus on sustainability storytelling, influencer partnerships, and high-engagement Reels content.',
 '["Instagram", "YouTube", "Google Search"]',
 'Vegan, zero-waste skincare built on transparency and trust.',
 '{"Q1": "Build Instagram presence", "Q2": "Launch influencer campaign", "Q3": "Run YouTube tutorials"}'),

(2,
 'Leverage content marketing, paid ads, and strategic webinars for B2B lead generation.',
 '["LinkedIn", "Google Ads", "Twitter"]',
 'Plug-and-play marketing AI for small businesses.',
 '{"Q1": "Establish LinkedIn presence", "Q2": "Launch Google Ads", "Q3": "Automate reporting workflows"}');

-- COMPETITORS
INSERT INTO competitors (business_id, name, keywords, ad_style, engagement_rate, notes)
VALUES
(1, 'GreenGlow', '["vegan serum", "eco skincare"]', 'Minimalist visuals, green palette', 4.8, 'Strong on influencer collabs'),
(1, 'NatureVibe', '["organic face oil"]', 'Natural imagery, emotional tone', 3.9, 'Frequent posts, moderate reach'),
(2, 'DataPulse', '["AI marketing dashboard"]', 'Professional tone, data visual-heavy', 5.2, 'High CPC but effective keywords');

-- CONTENTS
INSERT INTO contents (business_id, type, title, body, hashtags, platform)
VALUES
(1, 'blog', 'Top 5 Vegan Ingredients for Radiant Skin', 
 'Discover the natural power of vegan skincare with EcoBloom''s plant-based formulas.',
 '["#VeganSkincare", "#EcoBeauty", "#GlowNaturally"]', 'website'),

(1, 'social', 'Meet the Face Behind EcoBloom', 
 'Our founder shares why sustainable beauty matters.', 
 '["#FounderStory", "#SustainableSkincare", "#EcoBloom"]', 'instagram'),

(2, 'blog', 'How AI is Changing Marketing for Startups', 
 'TechSpark explains how automation can cut costs by 40% in early-stage startups.', 
 '["#AIMarketing", "#SaaSStartup", "#GrowthHacks"]', 'linkedin');

-- CAMPAIGNS
INSERT INTO campaigns (business_id, platform, objective, headlineA, headlineB, bodyA, bodyB, cta, status)
VALUES
(1, 'Instagram', 'Brand Awareness', 
 'Your Skin, Your Planet', 'Pure Beauty, Pure Earth',
 'EcoBloom brings skincare that cares for you and the planet.', 
 'Feel good knowing your glow is guilt-free.', 
 'Shop Now', 'active'),

(2, 'Google Ads', 'Lead Generation', 
 'Automate Your Marketing in Minutes', 'AI-Powered Analytics for Startups',
 'Try TechSpark''s all-in-one AI suite to supercharge your marketing.', 
 'Reduce costs, boost conversions - start your free trial today.',
 'Start Free Trial', 'draft');

-- KPIS
INSERT INTO kpis (business_id, date, impressions, clicks, conversions, ad_spend, revenue, roas, ctr, crr)
VALUES
(1, '2025-10-20', 25000, 2300, 180, 350.00, 1750.00, 5.0, 0.092, 0.22),
(2, '2025-10-20', 48000, 3800, 290, 820.00, 4300.00, 5.24, 0.079, 0.31);

-- TRENDS
INSERT INTO trends (category, source, topic, forecast_score, date_detected)
VALUES
('Skincare', 'GoogleTrends', 'vegan skincare', 0.87, '2025-10-26'),
('SaaS', 'Twitter', 'AI marketing tools', 0.91, '2025-10-26');

-- ENGAGEMENT SCRIPTS
INSERT INTO engagement_scripts (business_id, scenario, message_template, loyalty_idea)
VALUES
(1, 'new customer welcome', 
 'Welcome to EcoBloom - where your beauty meets sustainability! Enjoy 10% off your first order.', 
 'Eco Loyalty Points: Earn 1 point per $1 spent - redeem for future purchases.'),

(2, 'demo follow-up', 
 'Thanks for trying TechSpark! How was your experience? Let us know - we''re constantly improving your AI tools.',
 'Referral Boost: Invite 3 friends to unlock 1 free month of premium features.');

-- ========================================
-- ✅ DATABASE INITIALIZATION COMPLETE
-- ========================================
