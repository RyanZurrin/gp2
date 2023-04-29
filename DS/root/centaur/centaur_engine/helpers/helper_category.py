import centaur_deploy.constants as const_deploy


class CategoryHelper:
    """
    Contains utility functions for category numbers and category names.
    """
    # CADx
    MINIMAL = 0
    LOW = 1
    INTERMEDIATE = 2
    HIGH = 3
    # CADt
    NOT_SUSPICIOUS = 0
    SUSPICIOUS = 1

    @classmethod
    def get_category_texts(cls, run_mode):
        """
        Gets a mapping between category numbers and category names for the specified run mode.
        :param run_mode: str. The Centaur run mode.
        :return: dict. A dictionary with category numbers as keys and category names as values.
        """
        if run_mode in [const_deploy.RUN_MODE_CADX, const_deploy.RUN_MODE_DEMO]:
            return {
                cls.MINIMAL: "MINIMAL",
                cls.LOW: "LOW",
                cls.INTERMEDIATE: "INTERMEDIATE",
                cls.HIGH: "HIGH"
            }
        elif run_mode == const_deploy.RUN_MODE_CADT:
            return {
                cls.NOT_SUSPICIOUS: "Saige-Q: ",
                cls.SUSPICIOUS: "Saige-Q: Suspicious"
            }
        else:
            raise NotImplementedError("Got unsupported run_mode {}".format(run_mode))

    @classmethod
    def get_category_abbreviations(cls, run_mode):
        """
        Returns an abbreviated version of each category's name for the specified run mode.
        :return: dict. A dictionary with category numbers as keys and abbreviated category names as values.
        """
        assert run_mode in [const_deploy.RUN_MODE_CADX, const_deploy.RUN_MODE_DEMO, const_deploy.RUN_MODE_CADT], 'Got unsupported run_mode {}'.format(run_mode)

        if run_mode in [const_deploy.RUN_MODE_CADX, const_deploy.RUN_MODE_DEMO]:
            return {
                cls.MINIMAL: "MIN",
                cls.LOW: "LO",
                cls.INTERMEDIATE: "INT",
                cls.HIGH: "HI"
            }
        else:
            return {
                cls.NOT_SUSPICIOUS: "",
                cls.SUSPICIOUS: "Suspicious",
            }


    @classmethod
    def get_category_text(cls, category, run_mode):
        """
        Get the category text for a given numeric category
        :param category: int or str. Category "number"
        :param run_mode: str. The Centaur run mode.
        :return: str. Category text
        """
        category = int(category)
        d = cls.get_category_texts(run_mode=run_mode)
        assert category in d, 'Got unknown category {} for run_mode {}'.format(category, run_mode)
        return d[category]

    @classmethod
    def get_category_abbreviation(cls, category, run_mode):
        """
        Get the category text abbreviation for a given numeric category
        :param category: int or str. Category "number"
        :param run_mode: str. The Centaur run mode.
        :return: str. Category text abbreviated
        """
        assert run_mode in [const_deploy.RUN_MODE_CADX, const_deploy.RUN_MODE_DEMO, const_deploy.RUN_MODE_CADT], 'Got unsupported run_mode {}'.format(run_mode)
        category = int(category)
        d = cls.get_category_abbreviations(run_mode=run_mode)
        assert category in d, "Unknown category: {}".format(category)
        return d[category]

    @classmethod
    def get_all_categories(cls, run_mode):
        """
        Get a list of all the categories for the specified run mode.
        :param run_mode: str. The Centaur run mode.
        :return: list. A list of all the categories for the run mode.
        """
        return list(cls.get_category_texts(run_mode=run_mode).keys())
