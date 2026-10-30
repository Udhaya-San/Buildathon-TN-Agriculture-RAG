"""Page styling for the Streamlit app (kept from the original UI)."""

CSS = """
<style>

.stApp {
    background-color: #f6f8f5;
}

.gov-header {
    background-color: #1b5e20;
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

.main-title {
    color: #0d5c1d;
    font-size: 42px;
    text-align: center;
    font-weight: bold;
}

.sub-title {
    color: #333333;
    text-align: center;
    font-size: 18px;
}

.answer-card {
    background-color: white;
    color: black;
    padding: 20px;
    border-radius: 10px;
    border-left: 8px solid #2e7d32;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.15);
}

.stButton > button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1b5e20;
}

</style>
"""

GOV_HEADER = """
<div class="gov-header">
    <h2>Government of Tamil Nadu</h2>
    <h3>Department of Agriculture &amp; Farmers Welfare</h3>
</div>
"""

TITLE = """
<div class="main-title">🌾 Tamil Nadu Agriculture Schemes Assistant</div>
<div class="sub-title">Search agriculture schemes, subsidies, crop insurance, organic farming and farmer welfare programmes</div>
"""
