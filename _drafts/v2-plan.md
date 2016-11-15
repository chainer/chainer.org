---
title: Plan of v2
layout: post
categories: Announcement
---

We are planning the first major update of Chainer!
It is currently scheduled for next March or April.

The distinction of the release levels (major, minor, and revision) was made in v1.6.0 and is written in the [API Compatibility Policy](http://docs.chainer.org/en/stable/compatibility.html).
We are following this policy to decide what can be merged to Chainer in each release.
It states that each minor release keeps the backward compatibility.
Since the start of this policy, we have been trying to follow this rule and keeping the compatibility as much as possible (even the current Chainer v1.17.0 can run the examples included in v1.5.1).

A major update breaks the backward compatibility in some way.
In the first major update, we will mostly keep the current structure of features, and only update the details of API designs to improve the usability and consistencies between APIs.
We are currently planning to make changes in v2 on the following topics (note: the actual content may be changed):

- API consistencies between related features
- Relationships between Link and Optimizer
- Handling of parameter-shape placeholder
- Transparency between Variables and arrays

Another big change is CuPy separation.
**We are planning to remove CuPy module from Chainer and make it into a package in a separate repository.**
This will change the way how to develop Chainer and CuPy, and also how to install them.

The version 2 will be developed in a different style from the usual minor updates:

- The development will run on v2 branch, and keep the master branch developing the currently running v1.
- The v2 branch will use a new CuPy repository.
- Issues and PRs for v2 will be labeled as "v2."
- We will continue the development of v1 before the release of v2 (including CuPy). At each minor release, features included in it will be ported to v2.
- Features whose APIs are changed in v2 will be deprecated, and if the change is critical, they will raise FutureWarning.
- The development of CuPy will continue in master branch of Chainer. Until the release of v2, we will keep the CuPy repository updated by porting the CuPy-related changes of v1 into the CuPy repository.

We are planning to make a pre-release before the major update.
An alpha version is planned to be made in next January or February.
There is currently no plan of another pre-release, but if needed, we will make a beta release in next March.

The main development of v1 will be stopped after the release of v2.
After that, v1 will be moved into a maintenance mode, in which we update it only for critical issues.
The maintenance will run at least for half a year.
If there will be still many demands on supporting v1, we may extend the maintenance period.
