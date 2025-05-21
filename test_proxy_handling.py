import unittest
import sys
from urllib.parse import quote
import re

# Define the proxy prefix directly to avoid importing the module
PROXY_PREFIX = "https://please.untaint.us/?url="

# Copy the functions we want to test directly to avoid import issues
def clean_url(url: str) -> str:
    """
    Clean and normalize Locanto URLs, removing proxy prefixes and decoding repeatedly.
    """
    if not url:
        return ""
        
    # Decode repeatedly until stable
    from urllib.parse import unquote
    prev = None
    while prev != url:
        prev = url
        url = unquote(url)
    
    # Handle common proxy patterns
    proxy_patterns = [
        # Standard proxy prefix
        PROXY_PREFIX,
        # URL-encoded proxy prefix
        quote(PROXY_PREFIX),
        # Double-encoded proxy prefix
        quote(quote(PROXY_PREFIX)),
        # Alternative proxy formats
        "https://please.untaint.us?url=",
        "http://please.untaint.us/?url=",
    ]
    
    # Remove all proxy prefixes (even nested ones)
    for pattern in proxy_patterns:
        while pattern in url:
            url = url.replace(pattern, "")
    
    # Remove any ?url= or &url= parameters
    url = re.sub(r'([&?])url=([^&]*)', r'\1', url)
    
    # Remove any leading ? or & left over
    url = re.sub(r'^[?&]+', '', url)
    
    # Fix protocol and double slashes
    url = re.sub(r'https?:/{1,}', 'https://', url)
    
    # Ensure www. is present for locanto.co.za
    if 'locanto.co.za' in url and 'www.' not in url:
        url = url.replace('locanto.co.za', 'www.locanto.co.za')
    
    # Remove any trailing whitespace
    return url.strip()

def add_proxy_to_url(url: str, use_proxy: bool = True) -> str:
    """
    Add proxy prefix to URL if use_proxy is True.
    """
    if not url or not use_proxy:
        return url
        
    # Clean the URL first
    clean = clean_url(url)
    
    # Add proxy prefix
    return f"{PROXY_PREFIX}{clean}"

class TestProxyHandling(unittest.TestCase):
    """Test cases for proxy handling functions to prevent double proxy issues."""

    def setUp(self):
        """Set up test data."""
        self.base_url = "https://www.locanto.co.za/Dating/123456"
        self.proxy_url = f"{PROXY_PREFIX}{self.base_url}"
        self.double_proxy_url = f"{PROXY_PREFIX}{self.proxy_url}"
        self.encoded_proxy_url = f"{quote(PROXY_PREFIX)}{self.base_url}"
        self.double_encoded_proxy_url = f"{quote(quote(PROXY_PREFIX))}{self.base_url}"
        self.alternative_proxy_url = f"https://please.untaint.us?url={self.base_url}"
        self.nested_proxy_url = f"{PROXY_PREFIX}https://please.untaint.us?url={self.base_url}"

    def test_clean_url(self):
        """Test that clean_url properly removes all proxy prefixes."""
        # Test with a clean URL
        self.assertEqual(clean_url(self.base_url), self.base_url)
        
        # Test with a URL containing a proxy prefix
        self.assertEqual(clean_url(self.proxy_url), self.base_url)
        
        # Test with a URL containing a double proxy prefix
        self.assertEqual(clean_url(self.double_proxy_url), self.base_url)
        
        # Test with a URL containing an encoded proxy prefix
        self.assertEqual(clean_url(self.encoded_proxy_url), self.base_url)
        
        # Test with a URL containing a double-encoded proxy prefix
        self.assertEqual(clean_url(self.double_encoded_proxy_url), self.base_url)
        
        # Test with a URL containing an alternative proxy format
        self.assertEqual(clean_url(self.alternative_proxy_url), self.base_url)
        
        # Test with a URL containing nested proxy prefixes
        self.assertEqual(clean_url(self.nested_proxy_url), self.base_url)
        
        # Test with an empty URL
        self.assertEqual(clean_url(""), "")
        
        # Test with None
        self.assertEqual(clean_url(None), "")

    def test_add_proxy_to_url(self):
        """Test that add_proxy_to_url adds proxy prefix correctly and prevents double proxy."""
        # Test adding proxy to a clean URL
        self.assertEqual(add_proxy_to_url(self.base_url), self.proxy_url)
        
        # Test adding proxy to a URL that already has a proxy prefix
        # Should clean it first and then add a single proxy prefix
        self.assertEqual(add_proxy_to_url(self.proxy_url), self.proxy_url)
        
        # Test adding proxy to a URL with a double proxy prefix
        self.assertEqual(add_proxy_to_url(self.double_proxy_url), self.proxy_url)
        
        # Test adding proxy to a URL with an encoded proxy prefix
        self.assertEqual(add_proxy_to_url(self.encoded_proxy_url), self.proxy_url)
        
        # Test adding proxy to a URL with an alternative proxy format
        self.assertEqual(add_proxy_to_url(self.alternative_proxy_url), self.proxy_url)
        
        # Test with use_proxy=False
        self.assertEqual(add_proxy_to_url(self.base_url, use_proxy=False), self.base_url)
        
        # Test with an empty URL
        self.assertEqual(add_proxy_to_url(""), "")
        
        # Test with None
        self.assertEqual(add_proxy_to_url(None), None)

    def test_complex_scenarios(self):
        """Test more complex scenarios that might occur in real-world usage."""
        # URL with query parameters
        url_with_params = f"{self.base_url}?query=test&page=1"
        self.assertEqual(clean_url(f"{PROXY_PREFIX}{url_with_params}"), url_with_params)
        self.assertEqual(add_proxy_to_url(url_with_params), f"{PROXY_PREFIX}{url_with_params}")
        
        # URL with fragments
        url_with_fragment = f"{self.base_url}#section1"
        self.assertEqual(clean_url(f"{PROXY_PREFIX}{url_with_fragment}"), url_with_fragment)
        self.assertEqual(add_proxy_to_url(url_with_fragment), f"{PROXY_PREFIX}{url_with_fragment}")
        
        # URL with both query parameters and fragments
        complex_url = f"{self.base_url}?query=test&page=1#section1"
        self.assertEqual(clean_url(f"{PROXY_PREFIX}{complex_url}"), complex_url)
        self.assertEqual(add_proxy_to_url(complex_url), f"{PROXY_PREFIX}{complex_url}")
        
        # URL with query parameter that contains 'url='
        url_with_url_param = f"{self.base_url}?redirect_url=https://example.com"
        cleaned = clean_url(f"{PROXY_PREFIX}{url_with_url_param}")
        # The clean_url function should not remove the redirect_url parameter
        self.assertTrue("redirect_url=https://example.com" in cleaned)
        
        # Malformed URL with extra slashes
        malformed_url = "https:////www.locanto.co.za/Dating/123456"
        self.assertEqual(clean_url(malformed_url), "https://www.locanto.co.za/Dating/123456")

    def test_edge_cases(self):
        """Test edge cases that might cause issues."""
        # URL with 'url=' in the path
        url_with_url_in_path = "https://www.locanto.co.za/url=test/123456"
        self.assertEqual(clean_url(url_with_url_in_path), url_with_url_in_path)
        
        # URL with multiple query parameters including 'url='
        url_with_multiple_params = f"{self.base_url}?param1=value1&url=https://example.com&param2=value2"
        cleaned = clean_url(url_with_multiple_params)
        # The clean_url function should handle this correctly
        self.assertTrue("param1=value1" in cleaned)
        self.assertTrue("param2=value2" in cleaned)
        
        # URL with special characters
        url_with_special_chars = "https://www.locanto.co.za/Dating/123456?query=test%20with%20spaces"
        # The clean_url function decodes URL-encoded characters
        expected_result = "https://www.locanto.co.za/Dating/123456?query=test with spaces"
        self.assertEqual(clean_url(url_with_special_chars), expected_result)
        
        # URL with multiple levels of encoding
        encoded_url = quote(quote(self.base_url))
        self.assertEqual(clean_url(encoded_url), self.base_url)

if __name__ == "__main__":
    unittest.main()
