import os
import pandas as pd
import requests
from requests_oauthlib import OAuth2Session
from cachetools import cached, TTLCache

class SchwabAPI:
    """
    Wrapper for Schwab market data API to fetch instrument data and select top-N liquid symbols.
    """
    AUTHORIZATION_BASE_URL = "https://api.schwab.com/public/symphony/oauth2/authorize"
    TOKEN_URL = "https://api.schwab.com/public/symphony/oauth2/token"
    INSTRUMENT_URL = "https://api.schwab.com/public/symphony/quotes/v1/instruments"

    def __init__(self, client_id=None, client_secret=None, redirect_uri=None, scope=None):
        # Load client credentials
        self.client_id = client_id or os.getenv('SCHWAB_API_KEY')
        self.client_secret = client_secret or os.getenv('SCHWAB_API_SECRET')
        if not self.client_id or not self.client_secret:
            raise ValueError("SCHWAB_API_KEY and SCHWAB_API_SECRET must be set in the environment")
        # Determine redirect URI for authorization code flow
        self.redirect_uri = redirect_uri or os.getenv('SCHWAB_REDIRECT_URI')
        if not self.redirect_uri:
            raise ValueError("SCHWAB_REDIRECT_URI must be set in the environment or provided to __init__")
        # Scopes for API access (comma-separated env var or list)
        self.scope = scope or os.getenv('SCHWAB_SCOPE', '').split(',')
        # Initialize OAuth2Session for Authorization Code Grant
        self.oauth = OAuth2Session(
            client_id=self.client_id,
            redirect_uri=self.redirect_uri,
            scope=self.scope
        )

    def get_authorization_url(self):
        """Generate authorization URL for user redirection."""
        auth_url, state = self.oauth.authorization_url(self.AUTHORIZATION_BASE_URL)
        return auth_url, state

    def fetch_token(self, authorization_response):
        """Exchange the authorization response URL for an access token."""
        token = self.oauth.fetch_token(
            token_url=self.TOKEN_URL,
            authorization_response=authorization_response,
            client_secret=self.client_secret
        )
        self.token = token
        return token

    @cached(TTLCache(maxsize=128, ttl=3600))
    def get_market_caps(self, symbols):
        """
        Fetch marketCap for given list of symbols.
        Returns dict: {symbol: marketCap_value}
        """
        params = {'symbols': ','.join(symbols)}
        resp = self.oauth.get(self.INSTRUMENT_URL, params=params)
        resp.raise_for_status()
        instruments = resp.json().get('instruments', [])
        caps = {}
        for inst in instruments:
            sym = inst.get('symbol')
            cap = inst.get('marketCap')
            caps[sym] = cap
        return caps

    def get_top_n_liquid(self, n=100, constituents_csv='data/sp500_constituents.csv'):
        """
        Load S&P 500 constituents from CSV, fetch market caps, and return top-n by cap.
        """
        df = pd.read_csv(constituents_csv)
        symbols = df['Symbol'].astype(str).tolist()
        caps = self.get_market_caps(symbols)
        # Sort by cap descending, None->0
        sorted_syms = sorted(caps.items(), key=lambda kv: kv[1] or 0, reverse=True)
        top_symbols = [sym for sym, _ in sorted_syms[:n]]
        return top_symbols 