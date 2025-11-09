# ============================================
# MarketingAdvantage Advanced Prompt Library
# ============================================

def data_driven_prompt(business_name, industry, goal):
    return f"""
You are a data-driven marketing strategist with expertise in marketing analytics and ROI optimization.

For {business_name} in the {industry} industry, create a comprehensive marketing plan focused on {goal}.

**Required Sections:**
1. **Predictive Audience Segmentation**
   - Demographic/psychographic clusters with buying propensity scores
   - Customer lifetime value projections by segment
   - Churn risk analysis and retention strategies

2. **Multi-Touch Attribution Model**
   - Recommended attribution model (linear, time-decay, position-based)
   - Channel contribution weighting for {goal}
   - Cross-device tracking implementation

3. **Marketing Mix Modeling**
   - Budget allocation across channels with expected ROI
   - Seasonal adjustment factors for {industry}
   - Elasticity curves for pricing and ad spend

4. **Advanced KPI Framework**
   - Leading vs lagging indicators dashboard
   - Predictive metrics for {goal} achievement
   - Statistical significance thresholds for optimization

Format with data tables, confidence intervals, and implementation timelines.
"""


def ai_automation_prompt(business_name, industry, goal):
    return f"""
As an AI marketing automation specialist, develop a cutting-edge marketing strategy for {business_name} in {industry}.

**Focus Areas:**
1. **Generative AI Implementation**
   - AI content personalization engines for different audience segments
   - Dynamic creative optimization for ad campaigns
   - Chatbot and conversational marketing flows for {goal}

2. **Predictive Customer Journey Mapping**
   - Next-best-action recommendations at each touchpoint
   - Churn prediction triggers and intervention tactics
   - Lookalike audience modeling using AI clustering

3. **Marketing Technology Stack**
   - AI-powered tools for SEO, PPC, and social media optimization
   - Marketing automation workflows for lead nurturing
   - Real-time bidding strategies for programmatic advertising

4. **Testing & Optimization Framework**
   - A/B testing calendar with statistical power analysis
   - Multi-armed bandit approaches for channel optimization
   - AI-driven creative testing methodologies

Include specific AI tools, implementation costs, and expected efficiency gains.
"""


def b2b_enterprise_prompt(business_name, industry, goal):
    return f"""
You are a B2B marketing director with Fortune 500 experience. Create an enterprise marketing plan for {business_name} targeting {industry}.

**Strategic Components:**
1. **Account-Based Marketing (ABM) Framework**
   - Tier 1, 2, 3 account segmentation strategy
   - Personalized buying committee outreach plans
   - Account scoring model based on fit and engagement

2. **Sales-Marketing Alignment**
   - Lead scoring model with sales input
   - SLA between marketing and sales teams
   - Closed-loop reporting for {goal} attribution

3. **Content Strategy for Complex Sales Cycles**
   - Thought leadership content calendar for 6-month nurture
   - Case study development plan with ROI calculations
   - Executive briefing materials for C-level engagement

4. **Channel Strategy for Long Sales Cycles**
   - LinkedIn ABM advertising tactics
   - Industry event and trade show roadmap
   - Partner co-marketing program development

Include revenue targets by account tier and sales cycle compression strategies.
"""


def omnichannel_prompt(business_name, industry, goal):
    return f"""
As a customer experience strategist, design an omnichannel marketing plan for {business_name} that delivers seamless customer journeys.

**Required Elements:**
1. **Customer Journey Orchestration**
   - Touchpoint mapping across online/offline channels
   - Personalization at scale across all interactions
   - Moment-of-truth optimization for key conversion points

2. **Channel Integration Strategy**
   - Data synchronization across web, mobile, social, physical
   - Consistent messaging and branding across touchpoints
   - Real-time customer profile updates and triggers

3. **Experience Measurement Framework**
   - Customer effort score tracking at key interactions
   - Net Promoter Score correlation with marketing activities
   - Customer satisfaction impact on customer lifetime value

4. **Loyalty and Advocacy Program**
   - Multi-tier customer loyalty structure
   - Referral program mechanics for {industry}
   - Community building and user-generated content strategies

Include customer journey maps, technology requirements, and experience KPIs.
"""


def growth_hacking_prompt(business_name, industry, goal):
    return f"""
You are a growth marketing expert specializing in viral loops and exponential growth. Design a growth marketing plan for {business_name}.

**Growth Mechanics:**
1. **Viral Coefficient Optimization**
   - Natural sharing triggers built into the {industry} product/service
   - Referral program design with viral math calculations
   - Network effects exploitation strategies

2. **Growth Hacking Funnel**
   - Acquisition: Unconventional channels for {industry}
   - Activation: "Aha moment" identification and acceleration
   - Retention: Habit-forming product features and engagement loops
   - Revenue: Pricing strategy experiments and optimization
   - Referral: Built-in sharing mechanics

3. **Rapid Experimentation System**
   - Weekly growth experiment calendar
   - Hypothesis-driven testing framework
   - Scalable growth playbooks for successful tests

4. **Data-Driven Scaling**
   - Unit economics calculation for each growth channel
   - Scalability assessment of acquisition strategies
   - Growth team structure and weekly rhythm

Include specific growth tactics, expected viral coefficients, and scalability timelines.
"""


def competitive_analysis_prompt(business_name, industry, goal):
    return f"""
As a competitive intelligence specialist, create a marketing strategy for {business_name} that systematically outperforms competitors in {industry}.

**Competitive Analysis Framework:**
1. **Competitor Benchmarking Matrix**
   - SWOT analysis of top 3 competitors' marketing strategies
   - Market share trends and growth rate comparisons
   - Marketing spend efficiency ratios by competitor

2. **Gap Analysis & Positioning**
   - White space opportunities in the competitive landscape
   - Unique value proposition refinement against competitors
   - Competitive advantage sustainability assessment

3. **Offensive Marketing Tactics**
   - Competitor customer targeting strategies with overlap analysis
   - Conversion rate optimization against competitor weaknesses
   - Market disruption opportunities through innovation

4. **Defensive Strategy**
   - Customer retention programs to prevent competitor poaching
   - Brand defense messaging and rapid response protocols
   - Intellectual property and trade secret protection

Include specific competitor names, market share data projections, and win/loss analysis frameworks.
"""


# ============================================
# Advanced Prompt Generator (Dynamic Selector)
# ============================================

def create_advanced_marketing_prompt(business_name, industry, goal, specialization=None):
    base_prompt = f"""
You are an elite marketing strategist AI. Generate a comprehensive marketing plan for '{business_name}' in the '{industry}' industry.
The main goal is '{goal}'.
    """

    specializations = {
        "data_driven": data_driven_prompt(business_name, industry, goal),
        "b2b_enterprise": b2b_enterprise_prompt(business_name, industry, goal),
        "growth_hacking": growth_hacking_prompt(business_name, industry, goal),
        "omnichannel": omnichannel_prompt(business_name, industry, goal),
        "competitive": competitive_analysis_prompt(business_name, industry, goal),
        "ai_automation": ai_automation_prompt(business_name, industry, goal)
    }

    specialization_text = specializations.get(specialization, data_driven_prompt(business_name, industry, goal))

    format_requirements = """
**Format Requirements:**
- Executive summary with 3 key strategic pillars
- Quarterly roadmap with milestones and deliverables
- Budget allocation with ROI projections
- Risk assessment and mitigation strategies
- KPI dashboard with leading and lagging indicators

Present results in structured sections with clear subheadings.
"""

    return base_prompt + specialization_text + format_requirements
