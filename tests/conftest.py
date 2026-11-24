"""
Project ASTRAEUS - Global Test Configuration & Secure Environment Initialization.
Author: Dev Mehta <devmehta060505@gmail.com>
"""
import os

# Set secure high-entropy test HMAC secret key for test suite execution
os.environ.setdefault(
    "ASTRAEUS_HMAC_SECRET_KEY",
    "ASTRAEUS_OBSERVATORY_TEST_SECRET_KEY_2026_ENTROPY_DEV_MEHTA_SECURE"
)
