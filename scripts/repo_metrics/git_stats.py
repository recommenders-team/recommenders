# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Code based on: https://github.com/miguelgfierro/codebase/blob/master/python/utilities/git_stats.py
#

import git
import os
import requests
import datetime
from functools import lru_cache
import shutil


END_POINT = "https://api.github.com/repos/"
SEARCH_END_POINT = "https://api.github.com/search/"
BASE_URL = "https://github.com/"


class Github:
    """Github stats class"""

    def __init__(self, token, git_url):
        """Initializer.
        Args:
            token (str): Github token.
            git_url (str): URL of github repository.
        """
        self.token = token
        self.git_url = git_url
        self.repo_name = self.git_url.split(BASE_URL)[1]
        self.api_url = END_POINT + self.repo_name
        self.headers = {"Authorization": "token " + self.token}

    @property
    @lru_cache()
    def general_stats(self):
        """General attributes and statistics of the repo.
        Returns:
            json: JSON with general stats.
        """
        r = requests.get(self.api_url, headers=self.headers)
        if r.ok:
            return r.json()
        else:
            return None

    @property
    def forks(self):
        """Get current number of forks.
        Returns:
            int: Number of forks.
        """
        return (
            self.general_stats["forks_count"]
            if self.general_stats is not None
            else None
        )

    @property
    @lru_cache()
    def open_issues(self):
        """Get current number of open issues.
        Returns:
            int: Number of issues.
        """
        url = (
            SEARCH_END_POINT
            + "issues?q=state%3Aopen+repo:"
            + self.repo_name
            + "+type%3Aissues"
        )
        r = requests.get(url, headers=self.headers)
        if r.ok:
            return r.json()["total_count"]
        else:
            return None

    @property
    @lru_cache()
    def open_pull_requests(self):
        """Get current number of open PRs.
        Returns:
            int: Number of PRs.
        """
        url = (
            SEARCH_END_POINT
            + "issues?q=state%3Aopen+repo:"
            + self.repo_name
            + "+type%3Apr"
        )
        r = requests.get(url, headers=self.headers)
        if r.ok:
            return r.json()["total_count"]
        else:
            return None

    @property
    def stars(self):
        """Get current number of stars.
        Returns:
            int: Number of stars.
        """
        return (
            self.general_stats["stargazers_count"]
            if self.general_stats is not None
            else None
        )

    @property
    def watchers(self):
        """Get current number of watchers.
        Returns:
            int: Number of watchers.
        """
        return (
            self.general_stats["watchers_count"]
            if self.general_stats is not None
            else None
        )

    @property
    def subscribers(self):
        """Get current number of subscribers.
        Returns:
            int: Number of watchers.
        """
        return (
            self.general_stats["subscribers_count"]
            if self.general_stats is not None
            else None
        )

    @property
    @lru_cache()
    def last_year_commit_frequency(self):
        """Get the commit frequency in every week of the last year.
        Returns:
            dict: Dictionary of 52 elements (1 per week) with the commits every day 
                (starting on Sunday), total commit sum and first day of the week.
        """
        r = requests.get(self.api_url + "/stats/commit_activity", headers=self.headers)
        if r.ok:
            resp = r.json()
        else:
            return None
        for id, item in enumerate(resp):
            week_str = datetime.datetime.fromtimestamp(item["week"]).strftime(
                "%Y-%m-%d"
            )
            resp[id]["week"] = week_str
        return resp

    @property
    @lru_cache()
    def top_ten_referrers(self):
        """Get the top 10 referrers over the last 14 days.
        Source: https://developer.github.com/v3/repos/traffic/#list-referrers
        Returns:
            json: JSON with referrer name, total number of references
                and unique number of references.
        """
        r = requests.get(
            self.api_url + "/traffic/popular/referrers", headers=self.headers
        )
        if r.ok:
            return r.json()
        else:
            return None

    @property
    def number_total_referrers(self):
        """Count the total number of references to the repo.
        Returns:
            int: Number.
        """
        return (
            sum(item["count"] for item in self.top_ten_referrers)
            if self.top_ten_referrers is not None
            else None
        )

    @property
    def number_unique_referrers(self):
        """Count the unique number of references to the repo.
        Returns:
            int: Number.
        """
        return (
            sum(item["uniques"] for item in self.top_ten_referrers)
            if self.top_ten_referrers is not None
            else None
        )

    @property
    @lru_cache()
    def top_ten_content(self):
        """Get the top 10 popular contents within the repo over the last 14 days.
        Source: https://developer.github.com/v3/repos/traffic/#list-paths
        Returns:
            json: JSON with the content link, total and unique views.
        """
        r = requests.get(self.api_url + "/traffic/popular/paths", headers=self.headers)
        if r.ok:
            return r.json()
        else:
            return None

    @property
    @lru_cache()
    def views(self):
        """Get the total number of views and breakdown per day or week for the 
        last 14 days. Timestamps are aligned to UTC midnight of the beginning of 
        the day or week. Week begins on Monday.
        Source: https://developer.github.com/v3/repos/traffic/#views
        Returns:
            json: JSON with daily views.
        """
        r = requests.get(self.api_url + "/traffic/views", headers=self.headers)
        if r.ok:
            return r.json()
        else:
            return None

    @property
    def number_total_views(self):
        """Total number of views over the last 14 days
        Returns:
            int: Views.
        """
        return self.views["count"] if self.views is not None else None

    @property
    def number_unique_views(self):
        """Unique number of views over the last 14 days
        Returns:
            int: Views.
        """
        return self.views["uniques"] if self.views is not None else None

    @property
    @lru_cache()
    def clones(self):
        """Get the total number of clones and breakdown per day or week for the last
        14 days. Timestamps are aligned to UTC midnight of the beginning of the day 
        or week. Week begins on Monday.
        Source: https://developer.github.com/v3/repos/traffic/#clones
        Returns:
            json: JSON with daily clones. 
        """
        r = requests.get(self.api_url + "/traffic/clones", headers=self.headers)
        if r.ok:
            return r.json()
        else:
            return None

    @property
    def number_total_clones(self):
        """Total number of clones over the last 14 days
        Returns:
            int: Clones.
        """
        return self.clones["count"] if self.clones is not None else None

    @property
    def number_unique_clones(self):
        """Unique number of clones over the last 14 days
        Returns:
            int: Clones.
        """
        return self.clones["uniques"] if self.clones is not None else None

    @property
    def repo_size(self):
        """Repo size in Mb
        Returns:
            int: Size.
        """
        return self.general_stats["size"] if self.general_stats is not None else None

    @property
    def creation_date(self):
        """Date of repository creation
        Returns:
            str: Date.
        """
        return (
            self.general_stats["created_at"] if self.general_stats is not None else None
        )

    @property
    @lru_cache()
    def languages(self):
        """Get the languages in the repo and the lines of code of each.
        Source: https://developer.github.com/v3/repos/#list-languages
        Returns:
            dict: Dictionary of languages and lines of code.
        """
        r = requests.get(self.api_url + "/languages", headers=self.headers)
        if r.ok:
            return r.json()
        else:
            return None

    @property
    def number_languages(self):
        """Number of different languages
        Returns:
            int: Number
        """
        return len(self.languages) if self.languages is not None else None

    @property
    def number_commits(self):
        """Get total number of commits.
        NOTE: There is no straightforward way of getting the commits with GitHub API
        https://blog.notfoss.com/posts/get-total-number-of-commits-for-a-repository-using-the-github-api/
        Returns:
            int: Number of commits.
        """
        if self._cloned_repo_dir() is None:
            return None
        os.chdir(self._cloned_repo_dir())
        resp = os.popen("git rev-list HEAD --count").read()
        resp = int(resp.split("\n")[0])
        os.chdir("..")
        return resp

    @property
    def number_contributors(self):
        """Count the total number of contributors, based on unique email addresses.
        Returns:
            int: Number of contributors.
        """
        if self._cloned_repo_dir() is None:
            return None
        os.chdir(self._cloned_repo_dir())
        resp = os.popen('git log --format="%aN" | sort -u | wc -l').read()
        os.chdir("..")
        resp = int(resp.split("\n")[0])
        return resp

    @property
    def number_branches(self):
        """Number of current remote branches.
        Returns:
            int: Number.
        """
        if self._cloned_repo_dir() is None:
            return None
        os.chdir(self._cloned_repo_dir())
        resp = os.popen("git ls-remote --heads origin | wc -l").read()
        os.chdir("..")
        resp = int(resp.split("\n")[0])
        return resp

    @property
    def number_tags(self):
        """Number of tags.
        Returns:
            int: Number.
        """
        if self._cloned_repo_dir() is None:
            return None
        os.chdir(self._cloned_repo_dir())
        resp = os.popen("git tag | wc -l").read()
        os.chdir("..")
        resp = int(resp.split("\n")[0])
        return resp

    @property
    def number_total_lines(self):
        """Count total number of lines.
        Returns:
            int: Number of lines.
        """
        return sum(self.languages.values()) if self.languages is not None else None

    @property
    def number_added_lines(self):
        """Count the number of added lines.
        Returns:
            int: Number of added lines.
        """
        if self._cloned_repo_dir() is None:
            return None
        os.chdir(self._cloned_repo_dir())
        resp = os.popen(
            "git log  --pretty=tformat: --numstat | awk '{ add += $1 } END { printf \"%s\",add }'"
        ).read()
        os.chdir("..")
        resp = int(resp)
        return resp

    @property
    def number_deleted_lines(self):
        """Get the number of deleted lines.
        Returns:
            int: Number of deleted lines.
        """
        if self._cloned_repo_dir() is None:
            return None
        os.chdir(self._cloned_repo_dir())
        resp = os.popen(
            "git log  --pretty=tformat: --numstat | awk '{ add += $1 ; subs += $2 } END { printf \"%s\",subs }'"
        ).read()
        os.chdir("..")
        resp = int(resp)
        return resp

    def clean(self):
        if self._cloned_repo_dir() is not None:
            shutil.rmtree(self._cloned_repo_dir(), ignore_errors=True)

    def _cloned_repo_dir(self):
        """Clone a git repo and returns the location.
        Returns:
            str: Name of the folder name of the repo.
        """
        repo_dir = self.git_url.split("/")[-1]
        if os.path.isdir(repo_dir):
            return repo_dir
        try:
            git.Repo.clone_from(self.git_url, repo_dir)
        except git.GitCommandError:
            # try with token in case it is a private repo
            private_url = (
                "https://"
                + self.token
                + ":x-oauth-basic@github.com/"
                + self.git_url.split(BASE_URL)[1]
            )
            git.Repo.clone_from(private_url, repo_dir)
        if os.path.isdir(repo_dir):
            return repo_dir
        else:
            return None
