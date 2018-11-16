# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import traceback
import logging
from pymongo import MongoClient
from datetime import datetime
from scripts.repo_metrics.git_stats import Github
from scripts.repo_metrics.config import (
    GITHUB_TOKEN,
    CONNECTION_STRING,
    DATABASE,
    COLLECTION_GITHUB_STATS,
    COLLECTION_EVENTS,
    LOG_FILE,
)

logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)
format_str = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)s]: %(message)s"
format_time = "%Y-%m-%d %H:%M:%S"
logging.Formatter(format_str, format_time)
log = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Metrics Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("repo", type=str, help="GitHub repository")
    parser.add_argument(
        "--github_stats", action="store_true", help="Get today's repo statistics"
    )
    parser.add_argument(
        "--event",
        type=str,
        help="Input a general event that can be saved to the database",
    )
    parser.add_argument(
        "--save_to_database",
        action="store_true",
        help="Whether or not to save the information in the database",
    )
    return parser.parse_args()


def connect(uri="mongodb://localhost"):
    client = MongoClient(uri, serverSelectionTimeoutMS=1000)

    # Send a query to the server to see if the connection is working.
    try:
        client.server_info()
    except Exception:
        raise
    return client


def now():
    return datetime.now().strftime("%b %d %Y %H:%M:%S")


def event_as_dict(event):
    return {"date": now(), "event": event}


def github_stats_as_dict(github):
    return {
        "date": now(),
        "stars": github.stars,
        "forks": github.forks,
        "watchers": github.watchers,
        "open_issues": github.open_issues,
        "unique_views": github.number_unique_views,
        "total_views": github.number_total_views,
        "details_views": github.views,
        "unique_clones": github.number_unique_clones,
        "total_clones": github.number_total_clones,
        "details_clones": github.clones,
        "last_year_commit_frequency": github.last_year_commit_frequency,
        "details_referrers": github.top_ten_referrers,
        "total_referrers": github.number_total_referrers,
        "unique_referrers": github.number_unique_referrers,
        "details_content": github.top_ten_content,
        "repo_size": github.repo_size,
        "commits": github.number_commits,
        "contributors": github.number_contributors,
        "branches": github.number_branches,
        "tags": github.number_tags,
        "total_lines": github.number_total_lines,
        "added_lines": github.number_added_lines,
        "deleted_lines": github.number_deleted_lines,
    }


def tracker(github, args):
    if args.github_stats:
        git_doc = github_stats_as_dict(github)
        log.info("GitHub stats -- {}".format(git_doc))

    if args.event:
        event_doc = event_as_dict(args.event)
        log.info("Event -- {}".format(event_doc))

    if args.save_to_database:
        cli = connect(CONNECTION_STRING)
        db = cli[DATABASE]
        if args.github_stats:
            db[COLLECTION_GITHUB_STATS].insert_one(git_doc)
        if args.event:
            db[COLLECTION_EVENTS].insert_one(event_doc)


if __name__ == "__main__":
    args = parse_args()
    g = Github(GITHUB_TOKEN, args.repo)
    try:
        tracker(g, args)
    except Exception as e:
        trace = traceback.format_exc()
        log.error("Traceback: {}".format(trace))
        log.error("Exception: {}".format(e))
    finally:
        g.clean()
