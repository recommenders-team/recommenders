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
from datetime import datetime
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

format_str = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)s]: %(message)s"
format_time = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(
    filename=LOG_FILE, level=logging.DEBUG, format=format_str, datefmt=format_time
)
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
        type=_valid_date,
        default="now",
        help="Date for an event (format: YYYY-MM-DD or now)",
    )
    return parser.parse_args()


def _valid_date(s):
    """ Source: https://stackoverflow.com/a/25470943/5620182 """
    try:
        if s == "now":
            return datetime.now()
        else:
            return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "{} is not a valid date, use YYYY-MM-DD or now".format(s)
        raise argparse.ArgumentTypeError(msg)


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


def tracker(args):
    """Main function to track metrics.
    Args:
        args (obj): Parsed arguments.
    """
    if args.github_repo:
        # if there is an env variable, overwrite it
        token = os.environ.get("GITHUB_TOKEN", GITHUB_TOKEN)
        g = Github(token, args.github_repo)
        g.clean()  # clean folder if it exists
        git_doc = github_stats_as_dict(g)
        log.info("GitHub stats -- {}".format(git_doc))

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
        log.info("Tracking data")
        tracker(args)
    except Exception as e:
        trace = traceback.format_exc()
        log.error("Traceback: {}".format(trace))
        log.error("Exception: {}".format(e))
    finally:
        log.info("Routine finished")
