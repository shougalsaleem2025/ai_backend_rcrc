"""Strict extraction for leave-overlap date requests; never invent a year."""
from __future__ import annotations
import re
from datetime import date

MONTHS = {
    'يناير':1,'jan':1,'january':1,'فبراير':2,'feb':2,'february':2,
    'مارس':3,'march':3,'mar':3,'أبريل':4,'ابريل':4,'apr':4,'april':4,
    'مايو':5,'may':5,'يونيو':6,'june':6,'jun':6,'يوليو':7,'july':7,'jul':7,
    'أغسطس':8,'اغسطس':8,'اوقست':8,'august':8,'aug':8,'سبتمبر':9,'september':9,'sep':9,
    'أكتوبر':10,'اكتوبر':10,'october':10,'oct':10,'نوفمبر':11,'november':11,'nov':11,
    'ديسمبر':12,'december':12,'dec':12,
}
DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹','01234567890123456789')
MONTH_PATTERN = '|'.join(sorted(map(re.escape, MONTHS), key=len, reverse=True))
SEP = r'(?:إلى|الى|حتى|لين|لغاية|to|[-–—])'

def parse_overlap_dates(message: str):
    """Return (start_iso,end_iso,issue), where issue is '', 'missing_year', 'invalid' or 'missing_range'."""
    raw = str(message or '').translate(DIGITS)
    # E.g. 20 إلى 24 أكتوبر 2026; 20 أكتوبر إلى 24 أكتوبر 2026
    m = re.search(rf'\b(\d{{1,2}})\s*(?:({MONTH_PATTERN})\s*)?{SEP}\s*(\d{{1,2}})\s*({MONTH_PATTERN})(?:\s*[,،]?\s*(20\d{{2}}))?',raw,re.I)
    if m:
        first, first_month, last, last_month, year = m.groups()
        if not year:
            return '', '', 'missing_year'
        month1 = MONTHS[(first_month or last_month).lower()]
        month2 = MONTHS[last_month.lower()]
        try:
            start = date(int(year),month1,int(first))
            end = date(int(year),month2,int(last))
            if end < start:
                return '', '', 'invalid'
            return start.isoformat(),end.isoformat(),''
        except ValueError:
            return '', '', 'invalid'
    # Numeric two-date range, day/month/year (or ISO year/month/day)
    m = re.search(r'(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2})\s*(?:إلى|الى|حتى|to|[–—])\s*(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2})',raw,re.I)
    if m:
        def parse_one(s):
            parts = list(map(int,re.split('[-/]',s)))
            y,mo,d = parts if parts[0]>1900 else (parts[2],parts[1],parts[0])
            return date(y,mo,d)
        try:
            a,b=parse_one(m[1]),parse_one(m[2])
            if b<a: return '', '', 'invalid'
            return a.isoformat(),b.isoformat(),''
        except ValueError:
            return '', '', 'invalid'
    return '', '', 'missing_range'
