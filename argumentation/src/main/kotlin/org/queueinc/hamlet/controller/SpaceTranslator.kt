package org.queueinc.hamlet.controller

import it.unibo.tuprolog.argumentation.core.dsl.arg2pScope
import it.unibo.tuprolog.core.Term
import it.unibo.tuprolog.solve.MutableSolver
import it.unibo.tuprolog.solve.SolveOptions
import it.unibo.tuprolog.solve.TimeDuration
import it.unibo.tuprolog.unify.Unificator
import org.queueinc.hamlet.toSklearnClass

object SpaceTranslator {

    private fun translateList(terms: List<Term>) =
        terms.map {
            when {
                it.isReal -> it.castToReal().value.toDouble().toBigDecimal().toPlainString()
                it.isTrue || it.isNumber || it.isFail -> it
                else -> "\"$it\"".replace("'", "")
            }
    }.toString()

    @JvmStatic
    private fun translateSpace(space: Term) =
        arg2pScope {
            space.castToList().toList().joinToString(",\n") { step ->
                Unificator.default.mgu(step, tupleOf(X, "choice", Z)).let { unifier ->
                    unifier[Z]!!.castToList().toList().map { operator ->
                        if (operator.isAtom) {
                            if (operator.toString() == "function_transformer") {
                                """
                                {
                                    "type" : "${operator.toSklearnClass()}"
                                }
                                """
                            } else "\"$operator\""
                        } else {
                            Unificator.default.mgu(operator, tupleOf(A, B)).let { unifier2 ->
                                unifier2[B]!!.castToList().toList().joinToString(",\n") { hyper ->
                                    Unificator.default.mgu(hyper, tupleOf(C, D, E)).let { unifier3 ->
                                        """
                                        "${unifier3[C].toString().replace("'", "")}" : {
                                           "${unifier3[D]}" : ${translateList(unifier3[E]!!.castToList().toList())}
                                        }
                                        """
                                    }
                                }.let {
                                    """
                                    {
                                        "type" : "${unifier2[A]!!.toSklearnClass()}"${if (it.isEmpty()) "" else ","}
                                        $it
                                    }
                                    """
                                }
                            }
                        }
                    }.let {
                        """
                        "${unifier[X]}" : {
                           "choice" : $it
                        }
                        """
                    }
                }
            }.let {
                """
                {
                    $it
                }
                """.replace("\\s".toRegex(), "")
            }
        }

    @JvmStatic
    private fun translateTemplates(templates: List<Term>) : String {

        val toList = { term: Term ->
            term.castToList().toList()
        }

        val mapTerm = { step: Term, target: Term ->
            if (target.isAtom) "\"${target.toSklearnClass()}\""
            else if (step.toString() == "prototype") toList(target).map { "\"${it.castToList().toList().joinToString("_")}\"" }.toString()
            else toList(target).map { "\"${it.toSklearnClass()}\"" }.toString()
        }

        val transform = { target: Term ->
            arg2pScope {
                target.castToList().toList().map { template ->
                    Unificator.default.mgu(template, tupleOf(A, B, C, D)).let { unifier ->
                        val comparators = toList(unifier[B]!!)
                        val operators = toList(unifier[C]!!)
                        toList(unifier[A]!!).mapIndexed { i, step ->
                            val check = step.toString() != "prototype"
                            """
                                "$step" : {${if (check) "\"type\": {" else ""}"${comparators[i]}": ${mapTerm(step, operators[i])} ${if (check) "}" else ""}}
                            """
                        }.joinToString(",\n").let {
                            """
                            {
                                ${it + if (it == "") "" else ","}
                                "classification": {"type": {"eq": "${unifier[D]!!.toSklearnClass()}"}}
                            }
                            """
                        }
                    }
                }
            }
        }
        return templates.flatMap(transform).toString()
            .replace("\\s".toRegex(), "")
    }

    @JvmStatic
    fun mineData(solver: MutableSolver) =
        arg2pScope {

            println("Exporting Space")

            val space = solver.solve("miner" call "fetch_complete_space"(X), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                .filter { it.isYes }
                .map { translateSpace(it.substitution[X]!!) }
                .first()

            println("Exporting Constraints")

            val templates = solver.solve(
                ("miner" call "fetch_mandatory"(X)) and
                        ("miner" call "fetch_forbidden"(Y)) and
                        ("miner" call "fetch_mandatory_order"(Z)) and
                        ("miner" call "fetch_forbidden_instances"(T)) and
                        ("miner" call "fetch_forbidden_pipelines"(A)),
                    SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                .filter { it.isYes }
                .map { translateTemplates(listOf(X, Y, Z, T, A).map { v -> it.substitution[v]!! }) }
                .first()

//            println("Exporting Instances")

//            val instances = solver.solve("miner" call "fetch_instance_base_components"(X, Y), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
//                .filter { it.isYes }
//                .map {
//                    println("Collected Raw Instances")
//                    instances(it.substitution[X]!!, it.substitution[Y]!!)
//                }
//                .first()

            arrayOf(space, templates, "[]")
        }

    @JvmStatic
    fun mineConfig(config: Config, solver: MutableSolver) : Config =
        arg2pScope {
            println("Exporting Config")

            val clean = { input: Term? ->
                input.toString().replace("'", "")
            }

            val number = { input: Term? ->
                input.toString().toDouble()
            }

            val sensitiveFeatures = solver.solve(
                ("miner" call "fetch_sensitive_features"(X)),
                SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                .filter { it.isYes }
                .map { it.substitution[X]!!.castToList().toList().map { clean(it) } }
                .first()

            config.copy(
                dataset = solver.solve("dataset"(X)).filter { it.isYes }.map { clean(it.substitution[X]) }.firstOrNull(),
                metric = solver.solve("metric"(X)).filter { it.isYes }.map { clean(it.substitution[X]) }.firstOrNull(),
                fairnessMetric = solver.solve("fairness_metric"(X)).filter { it.isYes }.map { clean(it.substitution[X]) }.firstOrNull(),
                sensitiveFeatures = sensitiveFeatures,
                performanceThresholds = solver.solve("performance_thresholds"(X,Y)).filter { it.isYes }
                    .map { Pair(number(it.substitution[X]), number(it.substitution[Y])) }
                    .firstOrNull() ?: config.performanceThresholds,
                fairnessThresholds = solver.solve("fairness_thresholds"(X,Y)).filter { it.isYes }
                    .map { Pair(number(it.substitution[X]), number(it.substitution[Y])) }
                    .firstOrNull() ?: config.fairnessThresholds,
                miningSupport = solver.solve("mining_support"(X)).filter { it.isYes }
                    .map { number(it.substitution[X]) }.firstOrNull() ?: config.miningSupport,
            )
        }

}

//    @JvmStatic
//    private fun translateInstances(instances: Term) =
//        arg2pScope {
//            instances.castToList().toList().map { instance ->
//                instance.castToList().castToList().toList().joinToString(",\n") { step ->
//                    if (Unificator.default.match(step, tupleOf("prototype", `_`))) {
//                        Unificator.default.mgu(step, tupleOf("prototype", A)).let {
//                            "\"prototype\" : \"${it[A]}\""
//                        }
//                    }
//                    else if(Unificator.default.match(step, tupleOf(D, tupleOf(E, F)))) {
//                        Unificator.default.mgu(step, tupleOf(D, tupleOf(E, F))).let {
//                            it[F]!!.castToList().toList().joinToString(",\n") { h ->
//                                Unificator.default.mgu(h, tupleOf(G, H)).let { hyper ->
//                                    "\"${hyper[G]}\" : ${if (hyper[H]!!.isTrue || hyper[H]!!.isNumber || hyper[H]!!.isFail) hyper[H] else "\"${hyper[H]}\"".replace("'", "")}"
//                                }
//                            }.let { hyperparameters ->
//                                """
//                                    "${it[D]}": {
//                                        "type": "${it[E]!!.toSklearnClass()}"${if (hyperparameters.isEmpty()) "" else ","}
//                                        $hyperparameters
//                                    }
//                                    """
//                            }
//                        }
//                    }
//                    else {
//                        Unificator.default.mgu(step, tupleOf(B, C)).let {
//                            """
//                                "${it[B]}": {
//                                    "type": "${it[C]!!.toSklearnClass()}"
//                                }
//                                """
//                        }
//                    }
//                }.let {
//                    """
//                           {
//                                $it
//                           }
//                        """
//                }
//            }.toString().replace("\\s".toRegex(), "")
//        }

//    @JvmStatic
//    fun instances(operators: Term, pipelines: Term) : String {
//        val operatorList = operators.castToList().toList()
//        val pipelinesList = pipelines.castToList().toList().map { it.castToList().toList() }
//
//        val checkStep = { operator: Term, step: Term ->
//            operator.toString().contains(step.toString())
//        }
//
//        val checkPipeline = { pipeline: Term ->
//            pipeline.toString().let {
//                it.contains("function_transformer") ||
//                        it.contains("classification") ||
//                        it.contains("prototype")
//            }
//        }
//
//        val mapPrototype = { pipeline: List<Term> ->
//            pipeline.joinToString("_") { it.toString().substringBefore(",").replace("(", "").trim() }
//        }
//
//        val instances = pipelinesList.flatMap { pipeline ->
//            pipeline.filter { step -> !checkPipeline(step) }.map { step ->
//                operatorList.filter { operator -> checkStep(operator, step)}
//            }.fold(listOf(pipeline)) { acc, set ->
//                acc.flatMap { pipe -> set.map { op -> pipe.map {
//                    if (checkStep(op, it)) op else it
//                } } }
//            }.map { elem ->
//                it.unibo.tuprolog.core.List.of(elem + listOf(Struct.parse("(prototype, ${mapPrototype(elem)})")))
//            }
//        }.let { elem ->
//            it.unibo.tuprolog.core.List.of(elem)
//        }
//
//        return translateInstances(instances)
//    }
