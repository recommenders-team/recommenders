# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import os

# Need to append a full path instead of relative path.
# This seems to be an issue from Azure DevOps command line task.
# NOTE this does not affect running directly in the shell.
sys.path.append(os.getcwd())
import argparse
import traceback
import logging
from dateutil.parser import isoparse
from pymongo import MongoClient
from datetime import datetime
from scripts.repo_metrics.git_stats import Github
from scripts.repo_metrics.config import (
    GITHUB_TOKEN,
    CONNECTION_STRING,
    DATABASE,
    COLLECTION_GITHUB_STATS,
    COLLECTION_EVENTS,
)

format_str = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)s]: %(message)s"
format_time = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=format_str, datefmt=format_time)
log = logging.getLogger()


def parse_args():
    """Argument parser.
    Returns:
        obj: Parser.
    """
    parser = argparse.ArgumentParser(
        description="Metrics Tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--github_repo", type=str, help="GitHub repository")
    parser.add_argument(
        "--event",
        type=str,
        help="Input a general event that can be saved to the database",
    )
    parser.add_argument(
        "--save_to_database",
        action="store_true",
        help="Whether or not to save the information to the database",
    )
    parser.add_argument(
        "--event_date",
        default=datetime.now().isoformat(),
        type=isoparse,
        help="Date for an event (format: YYYY-MM-DD)",
    )
    return parser.parse_args()


def connect(uri="mongodb://localhost"):
    """Mongo connector.
    Args:
        uri (str): Connection string.
    Returns:
        obj: Mongo client.
    """
    client = MongoClient(uri, serverSelectionTimeoutMS=1000)

    # Send a query to the server to see if the connection is working.
    try:
        client.server_info()
    except Exception:
        raise
    return client


def event_as_dict(event, date):
    """Encodes an string event input as a dictionary with the date.
    Args:
        event (str): Details of a event.
        date (datetime): Date of the event.
    Returns:
        dict: Dictionary with the event and the date.
    """
    return {"date": date.strftime("%b %d %Y %H:%M:%S"), "event": event}


def github_stats_as_dict(github):
    """Encodes Github statistics as a dictionary with the date.
    Args:
        obj: Github object.
    Returns:
        dict: Dictionary with Github details and the date.
    """
    return {
        "date": datetime.now().strftime("%b %d %Y %H:%M:%S"),
        "stars": github.stars,
        "forks": github.forks,
        "watchers": github.watchers,
        "subscribers": github.subscribers,
        "open_issues": github.open_issues,
        "open_pull_requests": github.open_pull_requests,
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


def tracker(args):
    """Main function to track metrics.
    Args:
        args (obj): Parsed arguments.
    """
    if args.github_repo:
        # if there is an env variable, overwrite it
        token = os.environ.get("GITHUB_TOKEN", GITHUB_TOKEN)
        g = Github(token, args.github_repo)
        git_doc = github_stats_as_dict(g)
        log.info("GitHub stats -- {}".format(git_doc))
        g.clean()

    if args.event:
        event_doc = event_as_dict(args.event, args.event_date)
        log.info("Event -- {}".format(event_doc))

    if args.save_to_database:
        # if there is an env variable, overwrite it
        connection = token = os.environ.get("CONNECTION_STRING", CONNECTION_STRING)
        cli = connect(connection)
        db = cli[DATABASE]
        if args.github_repo:
            db[COLLECTION_GITHUB_STATS].insert_one(git_doc)
        if args.event:
            db[COLLECTION_EVENTS].insert_one(event_doc)


if __name__ == "__main__":
    log.info("Starting routine")
    args = parse_args()
    try:
        log.info("Arguments: {}".format(args))
        tracker(args)
    except Exception as e:
        trace = traceback.format_exc()
        log.error("Traceback: {}".format(trace))
        log.error("Exception: {}".format(e))
    finally:
        log.info("Routine finished")
