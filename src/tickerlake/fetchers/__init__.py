"""Data source fetchers. Each implements fetch -> validate -> write."""

from tickerlake.fetchers.base import BaseFetcher, FetchResult

__all__ = ["BaseFetcher", "FetchResult"]
